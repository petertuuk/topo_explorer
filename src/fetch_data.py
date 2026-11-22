"""Fetch elevation rasters from USGS 3DEP (US) or OpenTopography (Global).

Handles four main workflows:
  1. USGS 3DEP (US coverage): free, high-resolution (10 m / 30 m), no API key.
  2. Request planning: requests are sized by *pixel count*, not area, so a
     big area is fetched at the resolution it will actually be displayed at
     (one or a few requests) instead of hundreds of tiny full-res tiles.
  3. Parallel tiles, mosaicking & caching into GeoTIFFs under data/.
  4. Local basemaps: fast decimated reads from local NOAA ETOPO files, using
     internal overviews (built once) so zoomed-out reads are near-instant.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
USGS_3DEP_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)

DEMO_API_KEY = "demoapikeyot2022"

USGS_DEMTYPES = ("USGS_13", "USGS_1")

# OpenTopography's per-request area caps. (USGS requests are limited by
# pixel size instead — see USGS_MAX_REQUEST_PX.)
AREA_LIMIT_KM2 = {
    "SRTMGL1": 400_000,      # ~30 m
    "AW3D30": 400_000,       # ~30 m
    "COP30": 400_000,        # ~30 m
    "NASADEM": 400_000,      # ~30 m
    "SRTMGL3": 3_800_000,    # ~90 m
    "COP90": 3_800_000,      # ~90 m
    "SRTM15Plus": 100_000_000,  # ~450 m
}

RESOLUTION_M = {
    "USGS_13": 10,
    "USGS_1": 30,
    "SRTMGL1": 30,
    "AW3D30": 30,
    "COP30": 30,
    "NASADEM": 30,
    "SRTMGL3": 90,
    "COP90": 90,
    "SRTM15Plus": 450,
}

# Native pixel size in degrees.
NATIVE_RES_DEG = {
    "USGS_13": 1 / 10800,   # 1/3 arc-second
    "USGS_1": 1 / 3600,     # 1 arc-second
    "SRTMGL1": 1 / 3600,
    "AW3D30": 1 / 3600,
    "COP30": 1 / 3600,
    "NASADEM": 1 / 3600,
    "SRTMGL3": 1 / 1200,
    "COP90": 1 / 1200,
    "SRTM15Plus": 1 / 240,
}

# Longest side, in pixels, that we aim to fetch. Anything beyond what the
# app displays (MAX_DISPLAY_DIM) is downloaded and then thrown away.
DEFAULT_TARGET_PX = 2400

# The 3DEP ImageServer advertises 8000x8000 max, but large requests have
# returned HTTP 500 (server buffer limits) in practice, so stay conservative.
USGS_MAX_REQUEST_PX = 2048
USGS_WORKERS = 6
OPENTOPO_WORKERS = 2  # OpenTopography is rate-limited; don't hammer it.

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

BASEMAP_SOURCES = {
    "etopo60s": {
        "url": (
            "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/"
            "60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
        ),
        "resolution_m": 1855,
        "approx_size_mb": 444,
    },
    "etopo30s": {
        "url": (
            "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/"
            "30s/30s_surface_elev_gtif/ETOPO_2022_v1_30s_N90W180_surface.tif"
        ),
        "resolution_m": 927,
        "approx_size_mb": 1500,
    },
}

Bbox = tuple[float, float, float, float]


def bbox_area_km2(bbox: Bbox) -> float:
    """Rough area of a lon/lat bbox in km^2."""
    west, south, east, north = bbox
    lat_mid = math.radians((south + north) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_mid)
    width_km = (east - west) * km_per_deg_lon
    height_km = (north - south) * km_per_deg_lat
    return abs(width_km * height_km)


def is_in_us(bbox: Bbox) -> bool:
    """Check if bbox roughly falls inside the US (CONUS, Alaska, Hawaii)."""
    west, south, east, north = bbox
    conus = (-125.0 <= west and east <= -66.5 and 24.5 <= south and north <= 49.5)
    alaska = (-180.0 <= west and east <= -129.0 and 51.0 <= south and north <= 71.5)
    hawaii = (-160.5 <= west and east <= -154.5 and 18.5 <= south and north <= 22.5)
    return conus or alaska or hawaii


def _needed_res_deg(bbox: Bbox, target_px: int) -> float:
    west, south, east, north = bbox
    return max(abs(east - west), abs(north - south)) / target_px


def pick_demtype(
    bbox: Bbox, prefer_high_res: bool = True, target_px: int = DEFAULT_TARGET_PX
) -> str:
    """Choose a dataset based on location and how much detail is visible.

    US bboxes always use USGS 3DEP: it's resampled server-side to whatever
    resolution we ask for, so a large area costs no more than a small one.

    Elsewhere, OpenTopography always returns native resolution, so pick the
    coarsest dataset that's still at least as fine as the display needs —
    there's no point downloading 30 m SRTM for a whole country that will be
    shown at ~1 km/px.
    """
    if is_in_us(bbox):
        return "USGS_13" if prefer_high_res else "USGS_1"

    ordered = ["SRTMGL1", "SRTMGL3", "SRTM15Plus"]  # finest -> coarsest
    if not prefer_high_res:
        ordered = ordered[1:]
    needed = _needed_res_deg(bbox, target_px)
    fine_enough = [d for d in ordered if NATIVE_RES_DEG[d] <= needed]
    return fine_enough[-1] if fine_enough else ordered[0]


def split_bbox(bbox: Bbox, max_area_km2: float) -> list[Bbox]:
    """Split a bbox into an n x n grid whose cells each fit max_area_km2."""
    area = bbox_area_km2(bbox)
    if area <= max_area_km2:
        return [bbox]

    n = math.ceil(math.sqrt(area / max_area_km2))
    west, south, east, north = bbox
    lon_step = (east - west) / n
    lat_step = (north - south) / n
    return [
        (
            west + i * lon_step,
            south + j * lat_step,
            west + (i + 1) * lon_step,
            south + (j + 1) * lat_step,
        )
        for i in range(n)
        for j in range(n)
    ]


@dataclass(frozen=True)
class Tile:
    bbox: Bbox
    width: int | None = None   # pixel size, USGS only (OpenTopo is native)
    height: int | None = None


@dataclass(frozen=True)
class RequestPlan:
    demtype: str
    source: str               # "USGS 3DEP" or "OpenTopography"
    tiles: tuple[Tile, ...]
    res_deg: float            # effective pixel size of the result, degrees
    width: int                # approx. size of the merged raster, pixels
    height: int

    @property
    def res_m(self) -> float:
        """Effective ground resolution (east-west, at the bbox's mid-latitude)."""
        west, south, east, north = self.tiles[0].bbox
        lat_mid = math.radians((south + north) / 2)
        return self.res_deg * 111_320 * max(math.cos(lat_mid), 0.01)


def _plan_usgs(bbox: Bbox, demtype: str, target_px: int) -> RequestPlan:
    """Tile on an exact pixel grid so tiles line up when mosaicked."""
    west, south, east, north = bbox
    width_deg, height_deg = east - west, north - south
    res = max(NATIVE_RES_DEG[demtype], _needed_res_deg(bbox, target_px))
    total_w = max(1, math.ceil(width_deg / res))
    total_h = max(1, math.ceil(height_deg / res))
    dx, dy = width_deg / total_w, height_deg / total_h

    nx = math.ceil(total_w / USGS_MAX_REQUEST_PX)
    ny = math.ceil(total_h / USGS_MAX_REQUEST_PX)
    col_edges = [round(i * total_w / nx) for i in range(nx + 1)]
    row_edges = [round(j * total_h / ny) for j in range(ny + 1)]

    tiles = []
    for c0, c1 in zip(col_edges, col_edges[1:]):
        for r0, r1 in zip(row_edges, row_edges[1:]):
            tiles.append(
                Tile(
                    bbox=(west + c0 * dx, north - r1 * dy, west + c1 * dx, north - r0 * dy),
                    width=c1 - c0,
                    height=r1 - r0,
                )
            )
    return RequestPlan(demtype, "USGS 3DEP", tuple(tiles), max(dx, dy), total_w, total_h)


def plan_request(
    bbox: Bbox, demtype: str | None = None, target_px: int = DEFAULT_TARGET_PX
) -> RequestPlan:
    """Work out exactly what fetch_dem() will request, without fetching."""
    demtype = demtype or pick_demtype(bbox, target_px=target_px)
    if demtype in USGS_DEMTYPES:
        return _plan_usgs(bbox, demtype, target_px)

    west, south, east, north = bbox
    res = NATIVE_RES_DEG.get(demtype, 1 / 3600)
    tiles = tuple(Tile(b) for b in split_bbox(bbox, AREA_LIMIT_KM2.get(demtype, 400_000)))
    return RequestPlan(
        demtype,
        "OpenTopography",
        tiles,
        res,
        max(1, round((east - west) / res)),
        max(1, round((north - south) / res)),
    )


def _cache_key(demtype: str, bbox: Bbox, size: tuple[int, int] | None = None) -> str:
    raw = f"{demtype}_{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
    if size is not None:
        raw += f"_{size[0]}x{size[1]}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{demtype}_{digest}"


def _get_with_retry(url: str, params: dict, timeout: int, attempts: int = 4) -> requests.Response:
    """GET with retries on timeouts / 5xx. Both services produce these under
    load — 3DEP in particular returns 502s while it renders a new area, then
    serves the same request in under a second once it's warm."""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
        else:
            if response.status_code < 500 or attempt == attempts:
                return response
        time.sleep(2**attempt)  # 2, 4, 8 s
    raise AssertionError("unreachable")


def _write_atomic(out_path: Path, content: bytes) -> None:
    """Write to a temp file then rename, so a failed or interrupted download
    never leaves a truncated file that later looks like a valid cache hit."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    tmp.write_bytes(content)
    os.replace(tmp, out_path)


def _download_usgs_3dep_tile(tile: Tile, out_path: Path, timeout: int = 120) -> Path:
    """Download one USGS 3DEP tile via the ArcGIS ImageServer exportImage API.

    The server resamples to exactly the requested size, so we ask for the
    resolution we need rather than native 10 m.
    """
    west, south, east, north = tile.bbox
    params = {
        "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{tile.width},{tile.height}",
        "format": "tiff",  # GeoTIFF output
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    response = _get_with_retry(USGS_3DEP_EXPORT_URL, params, timeout)

    if response.status_code != 200:
        raise RuntimeError(f"USGS 3DEP HTTP error ({response.status_code}): {response.text[:500]}")

    # TIFF headers start with 'II' (little-endian) or 'MM' (big-endian)
    if not response.content.startswith((b"II\x2a\x00", b"MM\x00\x2a")):
        try:
            err_msg = response.json().get("error", {}).get("message", response.text[:500])
        except Exception:  # noqa: BLE001
            err_msg = response.text[:500]
        raise RuntimeError(f"USGS 3DEP API returned non-TIFF response: {err_msg}")

    _write_atomic(out_path, response.content)
    return out_path


def _download_opentopo_tile(
    tile: Tile, demtype: str, api_key: str, out_path: Path, timeout: int = 120
) -> Path:
    """Download one OpenTopography tile (always native resolution)."""
    west, south, east, north = tile.bbox
    params = {
        "demtype": demtype,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = _get_with_retry(OPENTOPO_URL, params, timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenTopography request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    _write_atomic(out_path, response.content)
    return out_path


def fetch_dem(
    bbox: Bbox,
    demtype: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    progress_callback=None,
    target_px: int = DEFAULT_TARGET_PX,
) -> Path:
    """Fetch an elevation GeoTIFF covering bbox (USGS 3DEP or OpenTopography).

    Tiles are downloaded in parallel. progress_callback, if given, is only
    ever called from the calling thread (safe for Streamlit widgets).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_request(bbox, demtype, target_px)
    demtype = plan.demtype
    is_usgs = demtype in USGS_DEMTYPES
    api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY") or DEMO_API_KEY

    def _report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    merged_size = (plan.width, plan.height) if is_usgs else None
    merged_path = cache_dir / f"{_cache_key(demtype, bbox, merged_size)}.tif"
    if merged_path.exists() and not force:
        _report(f"Using cached data: {merged_path.name}")
        return merged_path

    def _tile_path(tile: Tile) -> Path:
        size = (tile.width, tile.height) if is_usgs else None
        return cache_dir / f"{_cache_key(demtype, tile.bbox, size)}.tif"

    tile_paths = [_tile_path(t) for t in plan.tiles]
    todo = [
        (t, p) for t, p in zip(plan.tiles, tile_paths) if force or not p.exists()
    ]
    if todo:
        source = plan.source
        _report(f"Downloading {len(todo)} {demtype} tile(s) from {source}...")
        workers = USGS_WORKERS if is_usgs else OPENTOPO_WORKERS
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            futures = [
                pool.submit(_download_usgs_3dep_tile, t, p)
                if is_usgs
                else pool.submit(_download_opentopo_tile, t, demtype, api_key, p)
                for t, p in todo
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()  # re-raise any download error here
                _report(f"[{done}/{len(todo)}] tiles downloaded")

    if len(tile_paths) == 1:
        if tile_paths[0] != merged_path:
            _write_atomic(merged_path, tile_paths[0].read_bytes())
        return merged_path

    _report(f"Merging {len(tile_paths)} tiles...")
    _merge_tiles(tile_paths, merged_path)
    return merged_path


def _merge_tiles(tile_paths: list[Path], out_path: Path) -> None:
    import rasterio
    from rasterio.merge import merge

    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(sources)
        profile = sources[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
        tmp = out_path.with_name(out_path.name + ".part")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(mosaic)
        os.replace(tmp, out_path)
    finally:
        for src in sources:
            src.close()


def load_elevation(path: str | Path):
    """Load a GeoTIFF into a (data, extent) pair for plotting."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return data, extent


def basemap_path(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"basemap_{source}.tif"


def basemap_downloaded(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    return basemap_path(source, cache_dir).exists()


def basemap_has_overviews(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    import rasterio

    with rasterio.open(basemap_path(source, cache_dir)) as src:
        return bool(src.overviews(1))


def build_basemap_overviews(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
    """Add internal overviews (pre-shrunk copies) to a basemap GeoTIFF.

    One-time cost (~20 s for etopo60s, ~80 s for etopo30s; grows the file by
    ~40%). Afterwards, zoomed-out reads come from the matching overview level
    instead of decompressing the full-resolution data — e.g. CONUS from
    etopo30s goes from ~0.8 s to a few tens of ms.
    """
    import rasterio
    from rasterio.enums import Resampling

    path = basemap_path(source, cache_dir)
    with rasterio.Env(
        COMPRESS_OVERVIEW="DEFLATE", PREDICTOR_OVERVIEW=3, GDAL_NUM_THREADS="ALL_CPUS"
    ):
        with rasterio.open(path, "r+") as src:
            if not src.overviews(1):
                src.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)


def download_basemap(
    source: str = "etopo60s",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    progress_callback=None,
) -> Path:
    if source not in BASEMAP_SOURCES:
        raise KeyError(f"Unknown basemap source '{source}'. Options: {list(BASEMAP_SOURCES)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = basemap_path(source, cache_dir)

    if out_path.exists() and not force:
        return out_path

    url = BASEMAP_SOURCES[source]["url"]
    tmp_path = out_path.with_suffix(".tif.part")

    with requests.get(url, stream=True, timeout=60) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Basemap download failed ({response.status_code}) from {url}")
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    frac = downloaded / total if total else None
                    progress_callback(frac, downloaded)

    os.replace(tmp_path, out_path)
    build_basemap_overviews(source, cache_dir)
    return out_path


def read_basemap_window(
    bbox: Bbox,
    source: str = "etopo60s",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_dim: int | None = None,
):
    """Read bbox from a local basemap.

    If max_dim is given and the window is larger than that, the read is
    decimated on the fly (area-averaged) to fit, so we never decompress
    millions of pixels only to throw most of them away. With overviews
    present, GDAL reads from the closest pre-shrunk level automatically.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    path = basemap_path(source, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Basemap not downloaded yet: {path}. Call download_basemap() first."
        )

    west, south, east, north = bbox
    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()
        out_shape = None
        if max_dim and max(window.width, window.height) > max_dim:
            scale = max(window.width, window.height) / max_dim
            out_shape = (
                max(1, round(window.height / scale)),
                max(1, round(window.width / scale)),
            )
        data = src.read(
            1, window=window, out_shape=out_shape, resampling=Resampling.average
        ).astype(np.float32, copy=False)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        win_bounds = rasterio.windows.bounds(window, src.transform)
    extent = (win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3])
    return data, extent
