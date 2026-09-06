"""Fetch elevation rasters from USGS 3DEP (US) or OpenTopography (Global).

Handles four main workflows:
  1. USGS 3DEP (US Coverage): Free, high-resolution (10m / 30m), no API key, no rate limits.
  2. Area limits & Tiling: Automatically splits large bounding box requests.
  3. Mosaicking & Caching: Merges multi-tile responses into cached GeoTIFFs.
  4. Local Basemaps: Fast windowed reads from local NOAA ETOPO files.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import requests

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
USGS_3DEP_WCS_URL = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"

DEMO_API_KEY = "demoapikeyot2022"

AREA_LIMIT_KM2 = {
    "USGS_13": 50_000,       # ~10 m USGS 3DEP (1/3 arc-sec)
    "USGS_1": 200_000,       # ~30 m USGS 3DEP (1 arc-sec)
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


def bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough area of a lon/lat bbox in km^2."""
    west, south, east, north = bbox
    lat_mid = math.radians((south + north) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(lat_mid)
    width_km = (east - west) * km_per_deg_lon
    height_km = (north - south) * km_per_deg_lat
    return abs(width_km * height_km)


def is_in_us(bbox: tuple[float, float, float, float]) -> bool:
    """Check if bbox roughly falls inside the US (CONUS, Alaska, Hawaii)."""
    west, south, east, north = bbox
    conus = (-125.0 <= west and east <= -66.5 and 24.5 <= south and north <= 49.5)
    alaska = (-180.0 <= west and east <= -129.0 and 51.0 <= south and north <= 71.5)
    hawaii = (-160.5 <= west and east <= -154.5 and 18.5 <= south and north <= 22.5)
    return conus or alaska or hawaii


def pick_demtype(bbox: tuple[float, float, float, float], prefer_high_res: bool = True) -> str:
    """Choose dataset based on geographic area and resolution preference."""
    area = bbox_area_km2(bbox)
    
    # Prefer USGS 3DEP if inside US coordinates
    if is_in_us(bbox):
        if prefer_high_res and area <= AREA_LIMIT_KM2["USGS_13"]:
            return "USGS_13"
        elif area <= AREA_LIMIT_KM2["USGS_1"]:
            return "USGS_1"

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
    """Split a bbox into a grid fitting max_area_km2."""
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


import json

import json
import requests
from pathlib import Path

def _download_usgs_3dep_tile(
    bbox: tuple[float, float, float, float],
    demtype: str,
    out_path: Path,
    timeout: int = 120,
) -> Path:
    """Download USGS 3DEP elevation data via ArcGIS REST API."""
    west, south, east, north = bbox
    
    # 1/3 arc-sec (~10m) vs 1 arc-sec (~30m)
    res_deg = 0.00009259259 if demtype == "USGS_13" else 0.00027777777
    
    # Calculate pixel dimensions with a conservative cap (2048px) to avoid 500 buffer memory limits
    width = max(1, int(round(abs(east - west) / res_deg)))
    height = max(1, int(round(abs(north - south) / res_deg)))

    max_dim = 2048
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))

    # USGS 3DEP ImageServer export endpoint
    url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    
    # Rounded bounding box coordinates prevent internal geometry precision errors
    bbox_str = f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"

    params = {
        "bbox": bbox_str,
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "tiff",  # GeoTIFF output
        "pixelType": "F32",
        "noData": "-9999",
        "f": "image",
    }

    response = requests.get(url, params=params, timeout=timeout)
    
    if response.status_code != 200:
        raise RuntimeError(f"USGS 3DEP HTTP error ({response.status_code}): {response.text[:500]}")

    # TIFF headers start with 'II' (little-endian) or 'MM' (big-endian)
    if not response.content.startswith((b"II\x2a\x00", b"MM\x00\x2a")):
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", response.text[:500])
        except Exception:
            err_msg = response.text[:500]
        raise RuntimeError(f"USGS 3DEP API returned non-TIFF response: {err_msg}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return out_path

def _download_opentopo_tile(
    bbox: tuple[float, float, float, float],
    demtype: str,
    api_key: str,
    out_path: Path,
    timeout: int = 120,
) -> Path:
    """Download OpenTopography tile."""
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
    """Fetch elevation GeoTIFF covering bbox (USGS 3DEP or OpenTopography)."""
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
            if demtype in ("USGS_13", "USGS_1"):
                _report(f"[{idx}/{len(tiles)}] downloading {demtype} from USGS 3DEP...")
                _download_usgs_3dep_tile(tile_bbox, demtype, tile_path)
            else:
                _report(f"[{idx}/{len(tiles)}] downloading {demtype} tile from OpenTopography...")
                _download_opentopo_tile(tile_bbox, demtype, api_key, tile_path)
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
    """Load a GeoTIFF into a (data, extent) pair for plotting."""
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
    # data = np.clip(data, a_min=0, a_max=None)
    extent = (win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3])
    return data, extent