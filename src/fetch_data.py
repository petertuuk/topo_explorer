"""Fetch elevation rasters from OpenTopography's Global Datasets API.

Handles three things the raw API doesn't do for you:
  1. Area limits: each dataset caps how much area you can request in one
     call, so large bounding boxes are automatically split into a grid of
     sub-requests.
  2. Mosaicking: multiple tiles are merged back into a single GeoTIFF with
     rasterio.merge.
  3. Caching: results are cached on disk keyed by (demtype, bbox), so
     re-requesting the same area is instant and doesn't burn API calls.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import requests

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

# Public demo key: shared by everyone, capped at ~25 total requests. Fine for
# a first look; get your own free key from https://portal.opentopography.org
# and set OPENTOPOGRAPHY_API_KEY for real use.
DEMO_API_KEY = "demoapikeyot2022"

# Conservative per-request area caps (km^2), a bit under OpenTopography's
# documented limits so a request never gets rejected right at the edge.
AREA_LIMIT_KM2 = {
    "SRTMGL1": 400_000,      # ~30 m, documented cap 450,000
    "AW3D30": 400_000,       # ~30 m
    "COP30": 400_000,        # ~30 m
    "NASADEM": 400_000,      # ~30 m
    "SRTMGL3": 3_800_000,    # ~90 m, documented cap 4,050,000
    "COP90": 3_800_000,      # ~90 m
    "SRTM15Plus": 100_000_000,  # ~450 m, documented cap 125,000,000
}

# Roughly meters-per-pixel for each dataset, used to auto-pick a resolution
# based on how large an area is being requested.
RESOLUTION_M = {
    "SRTMGL1": 30,
    "AW3D30": 30,
    "COP30": 30,
    "NASADEM": 30,
    "SRTMGL3": 90,
    "COP90": 90,
    "SRTM15Plus": 450,
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------
# Bulk basemap: NOAA ETOPO 2022 global relief model. Single static files,
# no API key, no rate limit - download once, then every area is just a
# local rasterio window-read. Use this as your everyday data source; save
# the rate-limited OpenTopography API below for occasional high-res zooms.
# --------------------------------------------------------------------------
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


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough area of a lon/lat bbox in km^2 (equirectangular approximation,
    accurate enough for deciding how to tile a request)."""
    west, south, east, north = bbox
    lat_mid = math.radians((south + north) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_mid)
    width_km = (east - west) * km_per_deg_lon
    height_km = (north - south) * km_per_deg_lat
    return abs(width_km * height_km)


def pick_demtype(bbox: tuple[float, float, float, float], prefer_high_res: bool = True) -> str:
    """Choose the finest dataset whose area cap comfortably covers this bbox."""
    area = bbox_area_km2(bbox)
    if prefer_high_res:
        ordered = ["SRTMGL1", "SRTMGL3", "SRTM15Plus"]
    else:
        ordered = ["SRTMGL3", "SRTM15Plus"]
    for demtype in ordered:
        if area <= AREA_LIMIT_KM2[demtype]:
            return demtype
    return "SRTM15Plus"


def split_bbox(
    bbox: tuple[float, float, float, float], max_area_km2: float
) -> list[tuple[float, float, float, float]]:
    """Split a bbox into a grid of sub-bboxes that each fit under max_area_km2."""
    area = bbox_area_km2(bbox)
    if area <= max_area_km2:
        return [bbox]

    n = math.ceil(math.sqrt(area / max_area_km2))
    west, south, east, north = bbox
    lon_step = (east - west) / n
    lat_step = (north - south) / n

    tiles = []
    for i in range(n):
        for j in range(n):
            tiles.append(
                (
                    west + i * lon_step,
                    south + j * lat_step,
                    west + (i + 1) * lon_step,
                    south + (j + 1) * lat_step,
                )
            )
    return tiles


def _cache_key(demtype: str, bbox: tuple[float, float, float, float]) -> str:
    raw = f"{demtype}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{demtype}_{digest}"


def _download_tile(
    bbox: tuple[float, float, float, float],
    demtype: str,
    api_key: str,
    out_path: Path,
    timeout: int = 120,
) -> Path:
    west, south, east, north = bbox
    params = {
        "demtype": demtype,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = requests.get(OPENTOPO_URL, params=params, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"OpenTopography request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return out_path


def fetch_dem(
    bbox: tuple[float, float, float, float],
    demtype: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    progress_callback=None,
) -> Path:
    """Fetch a (possibly mosaicked) elevation GeoTIFF covering bbox.

    Parameters
    ----------
    bbox: (west, south, east, north) in decimal degrees.
    demtype: one of the OpenTopography dataset codes (SRTMGL1, SRTMGL3,
        SRTM15Plus, COP30, COP90, AW3D30, NASADEM). If None, a sensible
        resolution is picked automatically based on the bbox size.
    api_key: OpenTopography API key. Falls back to the
        OPENTOPOGRAPHY_API_KEY environment variable, then to the shared
        rate-limited demo key.
    cache_dir: where to store downloaded/merged GeoTIFFs.
    force: re-download even if a cached merged file already exists.
    progress_callback: optional callable(str) invoked with status messages,
        useful for updating a GUI while a multi-tile fetch is in progress.

    Returns
    -------
    Path to a single merged GeoTIFF covering bbox.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if demtype is None:
        demtype = pick_demtype(bbox)

    api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY") or DEMO_API_KEY

    merged_path = cache_dir / f"{_cache_key(demtype, bbox)}.tif"
    if merged_path.exists() and not force:
        if progress_callback:
            progress_callback(f"Using cached data: {merged_path.name}")
        return merged_path

    max_area = AREA_LIMIT_KM2.get(demtype, 400_000)
    tiles = split_bbox(bbox, max_area)

    def _report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    tile_paths = []
    for idx, tile_bbox in enumerate(tiles, start=1):
        tile_path = cache_dir / f"{_cache_key(demtype, tile_bbox)}.tif"
        if tile_path.exists() and not force:
            _report(f"[{idx}/{len(tiles)}] using cached tile")
        else:
            _report(f"[{idx}/{len(tiles)}] downloading {demtype} tile from OpenTopography...")
            _download_tile(tile_bbox, demtype, api_key, tile_path)
        tile_paths.append(tile_path)

    if len(tile_paths) == 1:
        single = tile_paths[0]
        if single != merged_path:
            merged_path.write_bytes(single.read_bytes())
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
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in sources:
            src.close()


def load_elevation(path: str | Path):
    """Load a GeoTIFF into a (data, extent) pair for imshow.

    extent is (west, east, south, north), matching matplotlib's convention.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return data, extent


def basemap_path(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"basemap_{source}.tif"


def basemap_downloaded(source: str = "etopo60s", cache_dir: str | Path = DEFAULT_CACHE_DIR) -> bool:
    return basemap_path(source, cache_dir).exists()


def download_basemap(
    source: str = "etopo60s",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force: bool = False,
    progress_callback=None,
) -> Path:
    """Download one of the static, unauthenticated, non-rate-limited global
    ETOPO 2022 GeoTIFFs (see BASEMAP_SOURCES). One-time cost of a few hundred
    MB to a couple GB; every subsequent area lookup is a local windowed read
    via read_basemap_window, with no further network calls.

    progress_callback, if given, is called as progress_callback(fraction)
    with fraction in [0, 1] (or None once total size is unknown) as bytes
    stream in.
    """
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

    tmp_path.rename(out_path)
    return out_path


def read_basemap_window(
    bbox: tuple[float, float, float, float],
    source: str = "etopo60s",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
):
    """Read just the portion of the downloaded basemap covering bbox.

    No network access - the basemap must already be downloaded via
    download_basemap(). Returns the same (data, extent) shape as
    load_elevation().
    """
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    path = basemap_path(source, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Basemap not downloaded yet: {path}. Call download_basemap() first."
        )

    west, south, east, north = bbox
    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, transform=src.transform)
        data = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        win_bounds = rasterio.windows.bounds(window, src.transform)
    extent = (win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3])  # west, east, south, north
    return data, extent
