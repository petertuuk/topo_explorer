"""Topo Explorer - Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import lru_cache

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LightSource, Normalize
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

def downsample_for_display(data: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    rows, cols = data.shape
    step = max(1, -(-max(rows, cols) // max_dim))  # ceil division
    if step == 1:
        return data

    # Sigma proportional to step size acts as an anti-aliasing low-pass filter
    smoothed = gaussian_filter(data.astype(np.float32), sigma=step / 2.0, mode="nearest")
    return smoothed[::step, ::step]


@st.cache_data(show_spinner=False)
def _load_state_boundaries():
    states = gpd.read_file(STATE_BOUNDARIES_URL).to_crs("EPSG:4326")
    # Simplify once here rather than on every render — the geometry is
    # far more detailed than what's ever visible on screen.
    states["geometry"] = states.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return states


def _boundary_xy_traces(gdf, bbox):
    """Vectorized clip + coordinate extraction shared by all boundary layers."""
    west, south, east, north = bbox
    subset = gdf.cx[west:east, south:north]
    if subset.empty:
        return [], []

    clipped = gpd.clip(subset, box(west, south, east, north))

    all_x, all_y = [], []
    for geometry in clipped.geometry:
        if geometry is None or geometry.is_empty:
            continue

        parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
        for part in parts:
            if part.is_empty:
                continue
            boundary = part.exterior if hasattr(part, "exterior") else part
            x, y = boundary.xy
            all_x.extend(list(x) + [None])
            all_y.extend(list(y) + [None])

    return all_x, all_y


def _add_state_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "state_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_state_boundaries(), bbox),
    )

    # Batch all geometries into one trace for optimal rendering speed
    if all_x:
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


@st.cache_data(show_spinner=False)
def _load_county_boundaries():
    counties = gpd.read_file(COUNTY_BOUNDARIES_URL).to_crs("EPSG:4326")
    # This is the big one (~3,200 polygons) — simplification here matters most.
    counties["geometry"] = counties.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return counties


def _add_county_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "county_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_county_boundaries(), bbox),
    )

    if all_x:
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


@st.cache_data(show_spinner=False)
def _load_lake_boundaries():
    lakes = gpd.read_file(LAKES_BOUNDARIES_URL).to_crs("EPSG:4326")
    lakes["geometry"] = lakes.geometry.simplify(
        BOUNDARY_SIMPLIFY_TOLERANCE, preserve_topology=False
    )
    return lakes


def _add_lake_boundaries(fig, bbox):
    all_x, all_y = _session_cache(
        "lake_boundary_trace_cache",
        st.session_state.get("bbox_key"),
        lambda: _boundary_xy_traces(_load_lake_boundaries(), bbox),
    )

    if all_x:
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


@st.cache_data(show_spinner=False)
def _load_cities():
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


@lru_cache(maxsize=None)
def matplotlib_to_plotly_colorscale(cmap_name: str, n_samples: int = 256) -> tuple:
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
        r, g, b, _ = [int(255 * c) for c in cmap(val)]
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

    c1, c2 = st.sidebar.columns(2)
    west = c1.number_input("West", format="%.3f", key="bbox_west")
    south = c2.number_input("South", format="%.3f", key="bbox_south")
    east = c1.number_input("East", format="%.3f", key="bbox_east")
    north = c2.number_input("North", format="%.3f", key="bbox_north")
    
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
    _resolved_demtype = demtype or fetch_data.pick_demtype(bbox)
    _area_km2 = fetch_data.bbox_area_km2(bbox)
    _resolution_m = fetch_data.RESOLUTION_M.get(_resolved_demtype)
    _area_limit_km2 = fetch_data.AREA_LIMIT_KM2.get(_resolved_demtype, 400_000)
    _tile_count = len(fetch_data.split_bbox(bbox, _area_limit_km2))
    _is_usgs = _resolved_demtype in ("USGS_13", "USGS_1")

    with st.sidebar.expander("📋 What will be requested", expanded=True):
        auto_note = (
            " (resolved by Auto)" if demtype_choice.startswith("Auto") else ""
        )
        st.write(f"**Dataset:** {_resolved_demtype}{auto_note}")
        st.write(
            f"**Source:** {'USGS 3DEP' if _is_usgs else 'OpenTopography'}  \n"
            f"**Resolution:** ~{_resolution_m} m/px"
        )
        st.write(
            f"**Area:** {_area_km2:,.0f} km²  \n"
            f"**Single-tile limit for this dataset:** {_area_limit_km2:,.0f} km²"
        )
        if _tile_count > 1:
            st.warning(
                f"This area exceeds the single-tile limit, so it will be split into "
                f"**{_tile_count} tiles**, downloaded one at a time and mosaicked "
                f"together. Large tile counts can take a while — consider a lower "
                f"resolution or a smaller area if this is too slow."
            )
        else:
            st.caption("Fits in a single tile request.")

st.sidebar.header("Display")
hillshade_on = st.sidebar.checkbox("Hillshade", value=True)
state_lines_on = st.sidebar.checkbox("State boundaries", value=True)
county_lines_on = st.sidebar.checkbox("County boundaries", value=False)
lake_lines_on = st.sidebar.checkbox("Lakes & inland water", value=False)
cities_on = st.sidebar.checkbox("Nearby cities (top 10 by population)", value=True)
colormap_choice = st.sidebar.selectbox("Colormap", list(COLORSCALES))

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False, persist="disk")
def _read_basemap_window(bbox, basemap_source):
    return fetch_data.read_basemap_window(bbox, source=basemap_source)


@st.cache_data(show_spinner=False, persist="disk")
def _fetch_and_load(bbox, demtype, api_key):
    path = fetch_data.fetch_dem(bbox, demtype=demtype, api_key=api_key or None)
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
    finite = data[np.isfinite(data)]
    st.session_state["elev_min"] = float(np.nanmin(finite)) if finite.size else 0.0
    st.session_state["elev_max"] = float(np.nanmax(finite)) if finite.size else 1.0

data = st.session_state["data"]
extent = st.session_state["extent"]
elev_min = st.session_state["elev_min"]
elev_max = st.session_state["elev_max"]

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


contour_on = st.sidebar.checkbox("Contour lines", value=False)
contour_interval = st.sidebar.slider(
    "Contour interval (meters)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
)

import contourpy


def _compute_contour_trace(display_data, bbox, interval=100):
    """Pure computation of contour line vertices — no fig, no Streamlit —
    so it can be cached and reused across reruns that don't change the
    DEM, extent, or interval (e.g. toggling an unrelated checkbox)."""
    west, south, east, north = bbox

    # 1. Mask missing or non-finite values
    valid_data = display_data[np.isfinite(display_data)]
    if valid_data.size == 0:
        return [], []

    min_elev = np.floor(valid_data.min() / interval) * interval
    max_elev = np.ceil(valid_data.max() / interval) * interval
    levels = np.arange(min_elev, max_elev + interval, interval)

    if len(levels) == 0:
        return [], []

    # 2. Build coordinate grids for contour calculation
    rows, cols = display_data.shape
    x_coords = np.linspace(west, east, cols)
    y_coords = np.linspace(north, south, rows)  # North to south

    # 3. Compute contour paths directly via contourpy (matplotlib's own
    # contouring backend) — no throwaway Figure/Axes needed just to get paths.
    # "Separate" line_type hands back one plain (N, 2) array per line segment,
    # so there's no path-code parsing (MOVETO/CLOSEPOLY) required either.
    # contourpy masks NaN/inf in z automatically, same as ax.contour did.
    gen = contourpy.contour_generator(
        x=x_coords, y=y_coords, z=display_data, line_type="Separate"
    )

    all_x, all_y = [], []
    for level in levels:
        for line in gen.lines(level):
            all_x.extend(line[:, 0].tolist())
            all_x.append(None)
            all_y.extend(line[:, 1].tolist())
            all_y.append(None)

    return all_x, all_y


def _add_contour_lines(fig, all_x, all_y):
    if all_x:
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

display_data = _session_cache(
    "display_data_cache",
    st.session_state.get("bbox_key"),
    lambda: downsample_for_display(data),
    label="downsample_for_display (cache miss)",
)
if display_data.shape != data.shape:
    st.sidebar.caption(
        f"Displaying downsampled to {display_data.shape[1]}×{display_data.shape[0]} px "
        "(full-res data is used for elevation calculations)."
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
    ]
    return smoothed_dem, intensity


if hillshade_on:
    # Keyed to the same bbox_key used for data invalidation, so this is only
    # recomputed when the DEM actually changes — not on every slider tweak.
    smoothed_dem, intensity = _session_cache(
        "hillshade_geometry",
        st.session_state.get("bbox_key"),
        lambda: _compute_hillshade_geometry(display_data),
        label="hillshade geometry (smooth+shade, cache miss)",
    )

    # Only depends on vmin/vmax/colormap — skipped when a rerun is triggered
    # by something unrelated (e.g. a boundary checkbox).
    def _compute_hillshade_color():
        cmap = plt.get_cmap(colorscale)
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(smoothed_dem))
        rgb = LIGHT_SOURCE.blend_soft_light(rgba[..., :3], intensity)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    rgb_img = _session_cache(
        "hillshade_color_cache",
        (st.session_state.get("bbox_key"), vmin, vmax, colorscale),
        _compute_hillshade_color,
        label="hillshade color mapping (cache miss)",
    )

    fig = go.Figure(
        go.Image(
            z=rgb_img,
            x0=west,
            dx=(east - west) / rgb_img.shape[1],
            y0=north,
            dy=-(north - south) / rgb_img.shape[0],
        )
    )
    fig.update_layout(yaxis=dict(autorange=True))
else:
    with _timed("heatmap build"):
        heatmap_colorscale = matplotlib_to_plotly_colorscale(colorscale)
        fig = go.Figure(
            go.Heatmap(
                z=display_data,
                x=np.linspace(west, east, display_data.shape[1]),
                y=np.linspace(north, south, display_data.shape[0]),
                colorscale=heatmap_colorscale,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="m"),
                hoverongaps=False,
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

with _timed("contour lines"):
    if contour_on:
        try:
            all_x, all_y = _session_cache(
                "contour_trace_cache",
                (st.session_state.get("bbox_key"), contour_interval),
                lambda: _compute_contour_trace(
                    display_data,
                    (west, south, east, north),
                    interval=contour_interval,
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
with st.sidebar.expander("⏱️ Timings (this rerun)", expanded=True):
    total_ms = sum(ms for _, ms in st.session_state["_timings"])
    for label, ms in st.session_state["_timings"]:
        st.write(f"{label}: **{ms:.1f} ms**")
    st.write(f"**Total measured: {total_ms:.1f} ms**")