"""Topo Explorer - Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import os
import time
from contextlib import contextmanager
from functools import lru_cache

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shapely
import streamlit as st
import contourpy
from matplotlib.colors import LightSource
from PIL import Image
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from src import fetch_data, regions

# Degrees of tolerance for simplifying boundary polygons before display.
# ~0.001 deg is well below pixel resolution at any zoom level we render,
# and cuts vertex counts (and clip/render cost) enormously for county-level data.
BOUNDARY_SIMPLIFY_TOLERANCE = 0.001

# --------------------------------------------------------------------------
# Lightweight profiling: wrap a section in `with _timed("label"):` and it
# records elapsed ms into st.session_state["_timings"] for this rerun.
# Cleared at the top of each script run; displayed at the bottom of the sidebar.
# --------------------------------------------------------------------------
st.session_state["_timings"] = []


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        st.session_state["_timings"].append((label, elapsed_ms))


def _session_cache(cache_name: str, key, compute, label: str | None = None):
    """Recompute `compute()` only when `key` changes since the last call for
    this `cache_name`; otherwise return the cached value. Unlike st.cache_data,
    this never hashes the inputs — you supply a cheap key yourself (e.g.
    bbox_key), so it's safe to use with large arrays as long as the key is
    something small that already identifies them (session state, not the array)."""
    cached_key, cached_value = st.session_state.get(cache_name, (None, None))
    if cached_key != key:
        if label:
            with _timed(label):
                cached_value = compute()
        else:
            cached_value = compute()
        st.session_state[cache_name] = (key, cached_value)
    return cached_value

st.set_page_config(page_title="Topo Explorer", layout="wide")
st.title("🗺️ Topo Explorer")
st.caption(
    "Elevation data from USGS 3DEP (US high-res), NOAA ETOPO 2022 (local offline), "
    "or OpenTopography. Drag the slider to stretch the colormap around terrain features."
)

DEMTYPE_LABELS = {
    "Auto (3DEP for US, OpenTopo for Global)": None,
    "USGS 3DEP (~10 m - US Only)": "USGS_13",
    "USGS 3DEP (~30 m - US Only)": "USGS_1",
    "SRTMGL1 (~30 m - Global)": "SRTMGL1",
    "COP30 (~30 m - Global)": "COP30",
    "SRTMGL3 (~90 m - Global)": "SRTMGL3",
    "COP90 (~90 m - Global)": "COP90",
    "SRTM15Plus (~450 m - Global)": "SRTM15Plus",
}

COLORSCALES = {
    "Earth tones": "gist_earth",
    "Viridis": "viridis",
    "Turbo": "turbo",
    "Greys": "gray",
}

# In land-only mode, colormaps whose low end is blue would make low-lying
# land look like water, so start them at their first "land" color instead.
LAND_CMAP_START = {"gist_earth": 0.3}

MAX_DISPLAY_DIM = 2400

STATE_BOUNDARIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/shp/cb_2025_us_state_500k.zip"
)
COUNTY_BOUNDARIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/shp/cb_2025_us_county_500k.zip"
)
LAKES_BOUNDARIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_lakes.geojson"
)
CITIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_populated_places_simple.geojson"
)
HIGHWAYS_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_roads.geojson"
)

VECTOR_CACHE_DIR = fetch_data.DEFAULT_CACHE_DIR / "vector"


def _read_vector_cached(cache_name: str, build):
    """Return a processed GeoDataFrame, building it (download + simplify +
    filter) only the first time and keeping the result in data/vector/ as
    GeoParquet. Without this, every Streamlit restart re-downloaded every
    layer (the roads file alone is tens of MB)."""
    path = VECTOR_CACHE_DIR / f"{cache_name}.parquet"
    if path.exists():
        try:
            return gpd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt/old cache: rebuild below
            pass
    gdf = build()
    try:
        VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(path)
    except Exception:  # noqa: BLE001 - e.g. pyarrow missing; caching is optional
        pass
    return gdf


def downsample_for_display(data: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    rows, cols = data.shape
    step = max(1, -(-max(rows, cols) // max_dim))  # ceil division
    if step == 1:
        return data

    # Sigma proportional to step size acts as an anti-aliasing low-pass filter
    smoothed = gaussian_filter(data.astype(np.float32), sigma=step / 2.0, mode="nearest")
    return smoothed[::step, ::step]


def _build_state_boundaries():
    states = gpd.read_file(STATE_BOUNDARIES_URL).to_crs("EPSG:4326")
    # Simplify once here rather than on every render — the geometry is
    # far more detailed than what's ever visible on screen.
    states["geometry"] = states.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return _to_lines(states[["geometry"]])


@st.cache_resource(show_spinner=False)
def _load_state_boundaries():
    return _read_vector_cached("states", _build_state_boundaries)


_EMPTY_XY = (np.empty(0), np.empty(0))


def _to_lines(gdf):
    """Convert polygon layers to their outlines (lines) once, at load time.

    Clipping outlines rather than polygons means the bbox edge never shows up
    as a fake boundary segment, and every layer can share the same fast
    line-extraction path below.
    """
    is_poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    if is_poly.any():
        gdf = gdf.copy()
        gdf.loc[is_poly, "geometry"] = gdf.loc[is_poly].geometry.boundary
    return gdf


def _lines_to_xy(geometries):
    """Flatten line geometries into one pair of x/y arrays with NaN gaps.

    Plotly draws a NaN as a break in the line, so a whole layer becomes a
    single trace. Returning numpy arrays (not Python lists with None) matters
    a lot: Plotly serializes arrays as compact binary, while lists of floats
    go through slow per-element JSON encoding on every rerun.
    """
    parts = shapely.get_parts(np.asarray(geometries, dtype=object))
    parts = parts[np.isin(shapely.get_type_id(parts), (1, 2))]  # LineString/LinearRing
    parts = parts[~shapely.is_empty(parts)]
    if parts.size == 0:
        return _EMPTY_XY
    coords, index = shapely.get_coordinates(parts, return_index=True)
    breaks = np.flatnonzero(np.diff(index)) + 1
    # float32 halves the payload; its ~1 m precision is far below pixel size.
    xy = np.insert(coords.astype(np.float32), breaks, np.nan, axis=0)
    return xy[:, 0], xy[:, 1]


def _boundary_xy_traces(gdf, bbox):
    """Clip a line layer to bbox and return NaN-separated x/y arrays."""
    west, south, east, north = bbox
    subset = gdf.cx[west:east, south:north]
    if subset.empty:
        return _EMPTY_XY
    clipped = gpd.clip(subset, box(west, south, east, north))
    return _lines_to_xy(clipped.geometry.values)


def _add_state_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "state_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_state_boundaries(), bbox),
    )

    # Batch all geometries into one trace for optimal rendering speed
    if len(all_x):
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y,
                mode="lines",
                line=dict(color="white", width=1.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _build_county_boundaries():
    counties = gpd.read_file(COUNTY_BOUNDARIES_URL).to_crs("EPSG:4326")
    # This is the big one (~3,200 polygons) — simplification here matters most.
    counties["geometry"] = counties.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return _to_lines(counties[["geometry"]])


@st.cache_resource(show_spinner=False)
def _load_county_boundaries():
    return _read_vector_cached("counties", _build_county_boundaries)


def _add_county_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "county_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_county_boundaries(), bbox),
    )

    if len(all_x):
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y,
                mode="lines",
                # Thin, dashed/semi-transparent lines to distinguish from state lines
                line=dict(color="rgba(255, 255, 255, 0.5)", width=0.75, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _build_lake_boundaries():
    lakes = gpd.read_file(LAKES_BOUNDARIES_URL).to_crs("EPSG:4326")
    lakes["geometry"] = lakes.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return _to_lines(lakes[["geometry"]])


@st.cache_resource(show_spinner=False)
def _load_lake_boundaries():
    return _read_vector_cached("lakes", _build_lake_boundaries)


def _add_lake_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "lake_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_lake_boundaries(), bbox),
    )

    if len(all_x):
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y,
                mode="lines",
                line=dict(color="#00ffff", width=1.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )


@st.cache_resource(show_spinner=False)
def _load_cities():
    return _read_vector_cached("cities", _build_cities)


def _build_cities():
    cities = gpd.read_file(CITIES_URL).to_crs("EPSG:4326")
    # Natural Earth's "simple" populated-places layer uses pop_max for the
    # largest reasonable population estimate per place; fall back gracefully
    # if the schema differs from what's expected.
    pop_col = next(
        (c for c in ("pop_max", "POP_MAX", "pop_min", "POP_MIN") if c in cities.columns),
        None,
    )
    name_col = next(
        (c for c in ("name", "NAME", "nameascii", "NAMEASCII") if c in cities.columns),
        None,
    )
    if pop_col is None or name_col is None:
        raise ValueError(
            "Unexpected cities dataset schema: missing name/population column"
        )
    return cities[[name_col, pop_col, "geometry"]].rename(
        columns={name_col: "name", pop_col: "population"}
    )


def _top_cities_in_bbox(bbox, top_n=10):
    west, south, east, north = bbox
    cities = _load_cities().cx[west:east, south:north]
    if cities.empty:
        return [], [], []

    top = cities.sort_values("population", ascending=False).head(top_n)
    return (
        top.geometry.x.tolist(),
        top.geometry.y.tolist(),
        top["name"].tolist(),
    )


def _add_city_markers(fig, bbox, top_n=10):
    xs, ys, names = _session_cache(
        "city_markers_cache",
        (st.session_state.get("bbox_key"), top_n),
        lambda: _top_cities_in_bbox(bbox, top_n=top_n),
    )

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=names,
                textposition="top center",
                textfont=dict(color="white", size=11),
                marker=dict(
                    symbol="circle",
                    size=7,
                    color="#ffcc00",
                    line=dict(color="black", width=1),
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )


@st.cache_resource(show_spinner=False)
def _load_highways():
    return _read_vector_cached("highways", _build_highways)


def _build_highways():
    roads = gpd.read_file(HIGHWAYS_URL).to_crs("EPSG:4326")

    # Natural Earth's roads layer covers every road class worldwide; filter
    # down to major highways only so this stays a lightweight reference
    # layer rather than a full road network. Schema naming has varied across
    # Natural Earth releases, so check a few likely column/value spellings
    # and fall back to the unfiltered set (still simplified) if none match.
    type_col = next(
        (c for c in ("type", "TYPE", "featurecla", "FEATURECLA") if c in roads.columns),
        None,
    )
    if type_col is not None:
        major_mask = roads[type_col].astype(str).str.contains(
            "Major Highway|Motorway|Beltway", case=False, na=False
        )
        if major_mask.any():
            roads = roads[major_mask]

    roads = roads[["geometry"]].copy()
    roads["geometry"] = roads.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return roads


def _add_highways(fig, bbox):
    all_x, all_y = _session_cache(
        "highways_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_highways(), bbox),
    )

    if len(all_x):
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y,
                mode="lines",
                line=dict(color="#ff6600", width=1.25),
                hoverinfo="skip",
                showlegend=False,
            )
        )



@lru_cache(maxsize=None)
def matplotlib_to_plotly_colorscale(
    cmap_name: str, n_samples: int = 256, start: float = 0.0
) -> tuple:
    """Converts a Matplotlib colormap into a Plotly-compatible colorscale.

    Pure function of its arguments (there are only a handful of colormap
    choices in this app), so cached rather than rebuilt on every rerun.
    Returns a tuple of tuples rather than a list of lists so the result is
    hashable/immutable — safe to share the cached object across calls.
    """
    cmap = plt.get_cmap(cmap_name)
    colorscale = []

    for i in range(n_samples):
        val = i / (n_samples - 1)
        r, g, b, _ = [int(255 * c) for c in cmap(start + (1 - start) * val)]
        colorscale.append((val, f"rgb({r},{g},{b})"))

    return tuple(colorscale)

def get_ot_api_key(filename: str = "OT_API_KEY") -> str:
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read().strip()
    return ""

# --------------------------------------------------------------------------
# Sidebar: area selection
# --------------------------------------------------------------------------
st.sidebar.header("Area")

region_names = sorted(regions.REGIONS)
region_choice = st.sidebar.selectbox(
    "Preset region", region_names, index=region_names.index("conus")
)

use_custom_bbox = st.sidebar.checkbox("Use custom bounding box instead")

if use_custom_bbox:
    def _pull_plot_extents():
        # Pull the selection dictionary from the Plotly state
        plot_state = st.session_state.get("topo_plot", {})
        selection = plot_state.get("selection", {})
        box = selection.get("box", [])
        
        if box:
            b = box[0]
            if "x" in b and "y" in b:
                st.session_state["bbox_west"] = float(min(b["x"]))
                st.session_state["bbox_east"] = float(max(b["x"]))
                st.session_state["bbox_south"] = float(min(b["y"]))
                st.session_state["bbox_north"] = float(max(b["y"]))

    st.sidebar.button(
        "🎯 Read Box Selection", 
        on_click=_pull_plot_extents, 
        use_container_width=True,
        help="Draw a rectangle on the map using the 'Box Select' tool, then click this button."
    )

    # Initialize widget states if they don't exist yet
    if "bbox_west" not in st.session_state:
        default_bbox = st.session_state.get("last_bbox", regions.REGIONS[region_choice])
        st.session_state["bbox_west"] = float(default_bbox[0])
        st.session_state["bbox_south"] = float(default_bbox[1])
        st.session_state["bbox_east"] = float(default_bbox[2])
        st.session_state["bbox_north"] = float(default_bbox[3])

    # A form, so the area is only loaded when you click the button — not
    # after every individual number you type (which, in online mode, meant
    # a download for each half-edited bbox).
    with st.sidebar.form("bbox_form", border=False):
        c1, c2 = st.columns(2)
        west = c1.number_input("West", format="%.3f", key="bbox_west")
        south = c2.number_input("South", format="%.3f", key="bbox_south")
        east = c1.number_input("East", format="%.3f", key="bbox_east")
        north = c2.number_input("North", format="%.3f", key="bbox_north")
        st.form_submit_button("Fetch this area", use_container_width=True)

    if west >= east or south >= north:
        st.sidebar.error("West must be less than East, and South less than North.")
        st.stop()
    bbox = (west, south, east, north)
else:
    bbox = regions.REGIONS[region_choice]

# --------------------------------------------------------------------------
# Sidebar: data source
# --------------------------------------------------------------------------
st.sidebar.header("Data source")
source_mode = st.sidebar.radio(
    "Where to get elevation data",
    ["Local basemap (offline)", "Online High-Res (USGS 3DEP / OpenTopo)"],
    index=0,
    help=(
        "Local basemap: NOAA ETOPO 2022 global model. Offline & fast.\n"
        "Online High-Res: USGS 3DEP (US 10m/30m, unlimited) or OpenTopography."
    ),
)

BASEMAP_LABELS = {
    "etopo30s": "etopo30s — ~1.5 GB, ~930 m/px",
    "etopo60s": "etopo60s — ~444 MB, ~1.8 km/px",
}

if source_mode == "Local basemap (offline)":
    basemap_source = st.sidebar.selectbox(
        "Basemap",
        list(BASEMAP_LABELS),
        format_func=lambda k: BASEMAP_LABELS[k],
    )

    if not fetch_data.basemap_downloaded(basemap_source):
        size_mb = fetch_data.BASEMAP_SOURCES[basemap_source]["approx_size_mb"]
        st.sidebar.warning(f"Basemap not downloaded yet (~{size_mb} MB, one-time).")
        if st.sidebar.button("Download basemap", type="primary", use_container_width=True):
            progress_bar = st.sidebar.progress(0.0)
            status = st.sidebar.empty()

            def _report_progress(frac, downloaded_bytes):
                mb = downloaded_bytes / 1e6
                if frac is not None:
                    progress_bar.progress(min(frac, 1.0))
                    status.caption(f"{mb:.0f} MB downloaded ({frac * 100:.0f}%)")
                else:
                    status.caption(f"{mb:.0f} MB downloaded")

            with st.spinner("Downloading basemap..."):
                fetch_data.download_basemap(
                    basemap_source, progress_callback=_report_progress
                )
            st.rerun()
        st.stop()

    if not fetch_data.basemap_has_overviews(basemap_source):
        st.sidebar.info(
            "One-time speed-up available: build overviews for this basemap so "
            "zoomed-out areas load in milliseconds instead of ~1 s."
        )
        if st.sidebar.button("Build overviews", use_container_width=True):
            with st.spinner("Building overviews (one time, ~1 minute)..."):
                fetch_data.build_basemap_overviews(basemap_source)
            st.rerun()
else:
    demtype_choice = st.sidebar.selectbox("Resolution / dataset", list(DEMTYPE_LABELS))
    demtype = DEMTYPE_LABELS[demtype_choice]

    api_key = get_ot_api_key()

    # --------------------------------------------------------------------
    # Request preview: show what will actually be fetched before doing it.
    # Especially useful for "Auto", since its US-vs-global cutoff is by
    # area, not just location — a bbox can be inside the US and still get
    # routed to a global dataset if it's too large for USGS 3DEP's limits.
    # --------------------------------------------------------------------
    _plan = fetch_data.plan_request(bbox, demtype, target_px=MAX_DISPLAY_DIM)
    _native_m = fetch_data.RESOLUTION_M.get(_plan.demtype)

    with st.sidebar.expander("📋 What will be requested", expanded=False):
        auto_note = (
            " (resolved by Auto)" if demtype_choice.startswith("Auto") else ""
        )
        st.write(f"**Dataset:** {_plan.demtype}{auto_note}")
        st.write(
            f"**Source:** {_plan.source}  \n"
            f"**Resolution:** ~{_plan.res_m:,.0f} m/px (native ~{_native_m} m)  \n"
            f"**Raster:** {_plan.width:,}×{_plan.height:,} px in "
            f"{len(_plan.tiles)} request(s)"
        )
        st.write(f"**Area:** {fetch_data.bbox_area_km2(bbox):,.0f} km²")
        if _plan.source == "USGS 3DEP":
            st.caption(
                "USGS resamples server-side to the resolution that's displayed, "
                "so zoom in (smaller box) for finer detail."
            )
        elif len(_plan.tiles) > 1:
            st.warning(
                f"OpenTopography caps request area, so this will be split into "
                f"**{len(_plan.tiles)} tiles** (downloaded in parallel). "
                f"Consider a coarser dataset or smaller area if this is slow."
            )

st.sidebar.header("Display")
hillshade_on = st.sidebar.checkbox("Hillshade", value=True)
land_only = st.sidebar.checkbox(
    "Flat-color oceans (land-only colors)",
    value=True,
    help=(
        "Draws everything below sea level as flat water and spends the whole "
        "color range on land, instead of half of it on ocean depth. Note: "
        "dry land below sea level (e.g. Death Valley) is drawn as water too."
    ),
)
state_lines_on = st.sidebar.checkbox("State boundaries", value=True)
county_lines_on = st.sidebar.checkbox("County boundaries", value=False)
lake_lines_on = st.sidebar.checkbox("Lakes & inland water", value=False)
cities_on = st.sidebar.checkbox("Nearby cities (top 10 by population)", value=True)
highways_on = st.sidebar.checkbox("Major highways", value=False)
colormap_choice = st.sidebar.selectbox("Colormap", list(COLORSCALES))

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


# In-memory only: the GeoTIFFs on disk already are the persistent cache, and
# persist="disk" pickled every (often 100+ MB) array into an ever-growing
# Streamlit cache folder.
@st.cache_data(show_spinner=False, max_entries=8)
def _read_basemap_window(bbox, basemap_source):
    return fetch_data.read_basemap_window(
        bbox, source=basemap_source, max_dim=MAX_DISPLAY_DIM
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _fetch_and_load(bbox, demtype, api_key):
    path = fetch_data.fetch_dem(
        bbox, demtype=demtype, api_key=api_key or None, target_px=MAX_DISPLAY_DIM
    )
    return fetch_data.load_elevation(path)


if source_mode == "Local basemap (offline)":
    state_key = ("basemap", basemap_source, bbox)
    need_load = "data" not in st.session_state or st.session_state.get("bbox_key") != state_key
    if need_load:
        with st.spinner("Reading area from local basemap..."):
            try:
                with _timed("data load (basemap read, cache miss)"):
                    data, extent = _read_basemap_window(bbox, basemap_source)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Read failed: {exc}")
                st.stop()
else:
    state_key = ("online_dem", demtype, bbox)
    need_load = (
        "data" not in st.session_state
        or st.session_state.get("bbox_key") != state_key
    )
    if need_load:
        with st.spinner(f"Fetching elevation data for {tuple(round(v, 2) for v in bbox)}..."):
            try:
                with _timed("data load (online fetch, cache miss)"):
                    data, extent = _fetch_and_load(bbox, demtype, api_key)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fetch failed: {exc}")
                st.stop()

if need_load:
    st.session_state["data"] = data
    st.session_state["extent"] = extent
    st.session_state["bbox_key"] = state_key
    st.session_state["last_bbox"] = bbox
    has_data = np.isfinite(data).any()
    st.session_state["elev_min"] = float(np.nanmin(data)) if has_data else 0.0
    st.session_state["elev_max"] = float(np.nanmax(data)) if has_data else 1.0
    # Lowest elevation at or above sea level, for the land-only color range.
    land = data[data >= 0]
    st.session_state["land_min"] = float(land.min()) if land.size else None

data = st.session_state["data"]
extent = st.session_state["extent"]
elev_max = st.session_state["elev_max"]
_land_min = st.session_state.get("land_min")
# Land-only only makes sense if the view actually contains land.
land_only = land_only and _land_min is not None
elev_min = _land_min if land_only else st.session_state["elev_min"]

# --------------------------------------------------------------------------
# Elevation range slider
# --------------------------------------------------------------------------
span = max(elev_max - elev_min, 1.0)
slider_lo = elev_min - 0.05 * span
slider_hi = elev_max + 0.05 * span

vmin, vmax = st.sidebar.slider(
    "Elevation range (m)",
    min_value=float(slider_lo),
    max_value=float(slider_hi),
    value=(float(elev_min), float(elev_max)),
    step=max(span / 200, 1.0),
)

st.sidebar.caption(
    f"Data range: {elev_min:.0f} m to {elev_max:.0f} m ({data.shape[1]}×{data.shape[0]} px)"
)


# --------------------------------------------------------------------------
# Contour lines slider
# --------------------------------------------------------------------------


CONTOUR_TARGET_LEVELS = 30


def _nice_contour_interval(elev_range: float, target_levels: int = CONTOUR_TARGET_LEVELS) -> int:
    """Round elev_range / target_levels up to a 1-2-5 x 10^n step (min 10 m),
    so a mountain range gets ~20-40 contours whether you're looking at one
    valley or the whole country."""
    raw = max(elev_range, 1.0) / target_levels
    magnitude = 10 ** np.floor(np.log10(raw))
    for multiple in (1, 2, 5, 10):
        if multiple * magnitude >= raw:
            return int(max(10, multiple * magnitude))
    return 10


contour_on = st.sidebar.checkbox("Contour lines", value=False)
contour_interval = None
if contour_on:
    auto_interval = _nice_contour_interval(elev_max - elev_min)
    contour_auto = st.sidebar.checkbox(
        "Automatic contour interval",
        value=True,
        help=f"Picks a round interval giving roughly {CONTOUR_TARGET_LEVELS} "
        "contour levels across the area's elevation range.",
    )
    if contour_auto:
        contour_interval = auto_interval
        n_levels = int((elev_max - elev_min) // auto_interval) + 1
        st.sidebar.caption(f"Contour interval: {auto_interval:,} m (~{n_levels} levels)")
    else:
        contour_interval = st.sidebar.slider(
            "Contour interval (meters)",
            min_value=10,
            max_value=1000,
            value=min(auto_interval, 1000),
            step=10,
        )


def _compute_contour_trace(display_data, bbox, interval=100, min_level=None):
    """Pure computation of contour line vertices — no fig, no Streamlit —
    so it can be cached and reused across reruns that don't change the
    DEM, extent, or interval (e.g. toggling an unrelated checkbox)."""
    west, south, east, north = bbox

    # 1. Mask missing or non-finite values
    valid_data = display_data[np.isfinite(display_data)]
    if valid_data.size == 0:
        return _EMPTY_XY

    min_elev = np.floor(valid_data.min() / interval) * interval
    if min_level is not None:
        min_elev = max(min_elev, min_level)
    max_elev = np.ceil(valid_data.max() / interval) * interval
    levels = np.arange(min_elev, max_elev + interval, interval)

    if len(levels) == 0:
        return _EMPTY_XY

    # 2. Build coordinate grids for contour calculation
    rows, cols = display_data.shape
    x_coords = np.linspace(west, east, cols)
    y_coords = np.linspace(north, south, rows)  # North to south

    # 3. Compute contour paths directly via contourpy (matplotlib's own
    # contouring backend) — no throwaway Figure/Axes needed just to get paths.
    # "Separate" line_type hands back one plain (N, 2) array per line segment,
    # so there's no path-code parsing (MOVETO/CLOSEPOLY) required either.
    # contourpy masks NaN/inf in z automatically, same as ax.contour did.
    # "ChunkCombinedNan" hands back each level's lines already joined into a
    # single array with NaN rows between them — exactly the shape Plotly
    # wants for one trace — so there's no per-line Python work at all.
    gen = contourpy.contour_generator(
        x=x_coords, y=y_coords, z=display_data, line_type="ChunkCombinedNan"
    )

    gap = np.array([[np.nan, np.nan]])
    pieces = []
    for level in levels:
        for chunk in gen.lines(level)[0]:
            if chunk is not None and len(chunk):
                pieces.append(chunk)
                pieces.append(gap)  # keep the next level from joining onto this one
    if not pieces:
        return _EMPTY_XY
    xy = np.concatenate(pieces).astype(np.float32)
    return xy[:, 0], xy[:, 1]


def _add_contour_lines(fig, all_x, all_y):
    if len(all_x):
        fig.add_trace(
            go.Scatter(
                x=all_x,
                y=all_y,
                mode="lines",
                line=dict(color="rgba(0, 0, 0, 0.4)", width=0.75),
                hoverinfo="skip",
                showlegend=False,
            )
        )

# --------------------------------------------------------------------------
# Figure Construction
# --------------------------------------------------------------------------
west, east, south, north = extent
colorscale = COLORSCALES[colormap_choice]
cmap_start = LAND_CMAP_START.get(colorscale, 0.0) if land_only else 0.0

display_data = _session_cache(
    "display_data_cache",
    st.session_state.get("bbox_key"),
    lambda: downsample_for_display(data),
    label="downsample_for_display (cache miss)",
)
if display_data.shape != data.shape:
    st.sidebar.caption(
        f"Displaying downsampled to {display_data.shape[1]}×{display_data.shape[0]} px."
    )

LIGHT_SOURCE = LightSource(azdeg=315, altdeg=45)
HILLSHADE_SMOOTH_SIGMA = 0.8
HILLSHADE_VERT_EXAG = 0.5


def _compute_hillshade_geometry(display_data: np.ndarray):
    """Terrain smoothing + shading intensity depend only on the DEM shape,
    not on the color slider or colormap — computed once per data load and
    reused across reruns triggered by those widgets."""
    finite = np.isfinite(display_data)
    baseline = float(np.nanmin(display_data)) if finite.any() else 0.0
    filled = np.where(finite, display_data, baseline)

    smoothed_dem = gaussian_filter(filled, sigma=HILLSHADE_SMOOTH_SIGMA)
    # Matches the shape matplotlib's shade_rgb() uses internally for blending.
    intensity = LIGHT_SOURCE.hillshade(smoothed_dem, vert_exag=HILLSHADE_VERT_EXAG)[
        ..., np.newaxis
    ].astype(np.float32)
    return smoothed_dem, intensity


NODATA_RGB = np.array([0.80, 0.80, 0.80], dtype=np.float32)  # light grey
WATER_RGB = np.array([0.66, 0.80, 0.90], dtype=np.float32)  # flat pale blue
HOVER_GRID_DIM = 500  # resolution of the invisible elevation-readout layer
JPEG_QUALITY = 90


@lru_cache(maxsize=None)
def _colormap_lut(cmap_name: str, start: float = 0.0) -> np.ndarray:
    """256-entry RGB lookup table for a colormap (float32, 0-1), optionally
    using only the part of the colormap from `start` up."""
    samples = np.linspace(start, 1, 256)
    return plt.get_cmap(cmap_name)(samples)[:, :3].astype(np.float32)


def _apply_colormap(
    values: np.ndarray, cmap_name: str, lo: float, hi: float, start: float = 0.0
) -> np.ndarray:
    """Map elevations to RGB via a 256-color lookup table.

    Same result as cmap(Normalize(lo, hi)(values)) at 8-bit precision, but
    several times faster because it's one uint8 index + gather in float32
    instead of matplotlib's general float64 path.
    """
    scale = 255.0 / max(hi - lo, 1e-6)
    idx = np.clip((values - lo) * scale, 0, 255)
    idx = np.nan_to_num(idx, nan=0).astype(np.uint8)
    rgb = _colormap_lut(cmap_name, start)[idx]
    nodata = ~np.isfinite(values)
    if nodata.any():
        rgb[nodata] = NODATA_RGB
    return rgb


def _encode_image_source(rgb_uint8: np.ndarray) -> str:
    """Encode an RGB array as a JPEG data URI for go.Image(source=...).

    Passing the raw pixel array (go.Image(z=...)) sent ~9 MB to the browser on
    every rerun for a full-size view, and made Plotly rebuild the image pixel
    by pixel in JavaScript. A JPEG is ~1 MB and the browser draws it natively.
    """
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _hover_grid(display_data: np.ndarray, bbox):
    """Coarse copy of the DEM for an invisible heatmap that provides the
    elevation hover readout and colorbar without shipping every pixel."""
    west, south, east, north = bbox
    step = max(1, -(-max(display_data.shape) // HOVER_GRID_DIM))
    z = display_data[step // 2 :: step, step // 2 :: step]
    rows, cols = display_data.shape
    dx, dy = (east - west) / cols, (north - south) / rows
    x = west + (np.arange(step // 2, cols, step) + 0.5) * dx
    y = north - (np.arange(step // 2, rows, step) + 0.5) * dy
    return z, x, y


if hillshade_on:
    # Keyed to the same bbox_key used for data invalidation, so this is only
    # recomputed when the DEM actually changes — not on every slider tweak.
    smoothed_dem, intensity = _session_cache(
        "hillshade_geometry",
        st.session_state.get("bbox_key"),
        lambda: _compute_hillshade_geometry(display_data),
        label="hillshade geometry (smooth+shade, cache miss)",
    )


def _compute_image_source():
    """Color the terrain (plus hillshade if on) and encode it as a JPEG.
    Only depends on the data, vmin/vmax, colormap, hillshade and land-only —
    skipped when a rerun is triggered by something unrelated."""
    if hillshade_on:
        rgb = _apply_colormap(smoothed_dem, colorscale, vmin, vmax, cmap_start)
        # Soft-light blend, same formula as matplotlib's
        # LightSource.blend_soft_light, done in float32.
        rgb = 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2
    else:
        rgb = _apply_colormap(display_data, colorscale, vmin, vmax, cmap_start)
    if land_only:
        rgb[display_data < 0] = WATER_RGB  # NaN compares False: stays nodata grey
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return _encode_image_source(rgb_uint8)


image_source = _session_cache(
    "image_source_cache",
    (st.session_state.get("bbox_key"), vmin, vmax, colorscale, hillshade_on, land_only),
    _compute_image_source,
    label="terrain image color+encode (cache miss)",
)
hover_z, hover_x, hover_y = _session_cache(
    "hover_grid_cache",
    st.session_state.get("bbox_key"),
    lambda: _hover_grid(display_data, (west, south, east, north)),
)

with _timed("figure build"):
    fig = go.Figure()
    # The terrain is a layout image pinned by its top-left corner to
    # (west, north) and stretched to the data extent. Unlike go.Image, whose
    # row order depends on dy's sign and the axis direction (and differs
    # between z= and source=), this can't come out flipped.
    fig.add_layout_image(
        source=image_source,
        xref="x",
        yref="y",
        x=west,
        y=north,
        sizex=east - west,
        sizey=north - south,
        xanchor="left",
        yanchor="top",
        sizing="stretch",
        layer="below",
    )
    fig.update_xaxes(range=[west, east], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[south, north], showgrid=False, zeroline=False)
    # Invisible, coarse elevation layer on top: gives the hover readout and
    # the colorbar (now in hillshade mode too) for a tiny payload.
    fig.add_trace(
        go.Heatmap(
            z=hover_z,
            x=hover_x,
            y=hover_y,
            colorscale=matplotlib_to_plotly_colorscale(colorscale, start=cmap_start),
            zmin=vmin,
            zmax=vmax,
            opacity=0,
            colorbar=dict(title="m"),
            hoverongaps=False,
            hovertemplate="%{y:.3f}°, %{x:.3f}°<br>%{z:,.0f} m<extra></extra>",
        )
    )

with _timed("state boundaries"):
    if state_lines_on:
        try:
            _add_state_boundaries(fig, (west, south, east, north))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load state boundaries: {exc}")

with _timed("county boundaries"):
    if county_lines_on:
        try:
            _add_county_boundaries(fig, (west, south, east, north))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load county boundaries: {exc}")

with _timed("lake boundaries"):
    if lake_lines_on:
        try:
            _add_lake_boundaries(fig, (west, south, east, north))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load lake boundaries: {exc}")

with _timed("city markers"):
    if cities_on:
        try:
            _add_city_markers(fig, (west, south, east, north), top_n=10)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load city markers: {exc}")

with _timed("highways"):
    if highways_on:
        try:
            _add_highways(fig, (west, south, east, north))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load highways: {exc}")

with _timed("contour lines"):
    if contour_on:
        try:
            all_x, all_y = _session_cache(
                "contour_trace_cache",
                (st.session_state.get("bbox_key"), contour_interval, land_only),
                lambda: _compute_contour_trace(
                    display_data,
                    (west, south, east, north),
                    interval=contour_interval,
                    min_level=0.0 if land_only else None,
                ),
                label="contour lines (cache miss)",
            )
            _add_contour_lines(fig, all_x, all_y)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not render contours: {exc}")

# Lock aspect ratio so geography isn't stretched on resize
fig.update_yaxes(scaleanchor="x", scaleratio=1)

# Change dragmode to "select" so users can draw a bounding box
fig.update_layout(
    height=750,
    margin=dict(l=0, r=0, t=10, b=0),
    dragmode="select",  # Changes default drag behavior to box selection
    xaxis_title="Longitude",
    yaxis_title="Latitude",
)

with _timed("st.plotly_chart render"):
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={"scrollZoom": True, "displaylogo": False,
            "modeBarButtonsToAdd": ["select2d"]},
        key="topo_plot",
        on_select="rerun"
    )

st.caption(
    "Scroll to zoom, drag to pan. "
    "USGS 3DEP datasets are auto-selected for US bounding boxes without rate limits."
)

# --------------------------------------------------------------------------
# Profiling readout — remove once you're done diagnosing.
# --------------------------------------------------------------------------
with st.sidebar.expander("⏱️ Timings (this rerun)", expanded=False):
    total_ms = sum(ms for _, ms in st.session_state["_timings"])
    for label, ms in st.session_state["_timings"]:
        st.write(f"{label}: **{ms:.1f} ms**")
    st.write(f"**Total measured: {total_ms:.1f} ms**")