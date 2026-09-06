"""Topo Explorer - Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LightSource, Normalize
from scipy.ndimage import gaussian_filter
from shapely.ops import clip_by_rect

from src import fetch_data, regions

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
    states = gpd.read_file(STATE_BOUNDARIES_URL)
    return states.to_crs("EPSG:4326")


def _add_state_boundaries(fig, bbox):
    west, south, east, north = bbox
    states = _load_state_boundaries().cx[west:east, south:north]

    all_x, all_y = [], []

    for geometry in states.geometry:
        if geometry is None or geometry.is_empty:
            continue

        # Cleanly slice boundaries at the exact bounding box edges
        clipped = clip_by_rect(geometry, west, south, east, north)
        if clipped.is_empty:
            continue

        parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]

        for part in parts:
            if part.is_empty:
                continue

            boundary = part.exterior if hasattr(part, "exterior") else part
            x, y = boundary.xy

            # Append path points followed by None to break the line between shapes
            all_x.extend(list(x) + [None])
            all_y.extend(list(y) + [None])

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
    counties = gpd.read_file(COUNTY_BOUNDARIES_URL)
    return counties.to_crs("EPSG:4326")


def _add_county_boundaries(fig, bbox):
    west, south, east, north = bbox
    counties = _load_county_boundaries().cx[west:east, south:north]

    all_x, all_y = [], []

    for geometry in counties.geometry:
        if geometry is None or geometry.is_empty:
            continue

        clipped = clip_by_rect(geometry, west, south, east, north)
        if clipped.is_empty:
            continue

        parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]

        for part in parts:
            if part.is_empty:
                continue

            boundary = part.exterior if hasattr(part, "exterior") else part
            x, y = boundary.xy

            all_x.extend(list(x) + [None])
            all_y.extend(list(y) + [None])

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


import osmnx as ox


import osmnx as ox


@st.cache_data(show_spinner=False)
def _load_lake_boundaries():
    lakes = gpd.read_file(LAKES_BOUNDARIES_URL)
    return lakes.to_crs("EPSG:4326")


def _add_lake_boundaries(fig, bbox):
    west, south, east, north = bbox
    lakes = _load_lake_boundaries().cx[west:east, south:north]

    all_x, all_y = [], []

    for geometry in lakes.geometry:
        if geometry is None or geometry.is_empty:
            continue

        clipped = clip_by_rect(geometry, west, south, east, north)
        if clipped.is_empty:
            continue

        parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]

        for part in parts:
            if part.is_empty:
                continue

            boundary = part.exterior if hasattr(part, "exterior") else part
            x, y = boundary.xy

            all_x.extend(list(x) + [None])
            all_y.extend(list(y) + [None])

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


def matplotlib_to_plotly_colorscale(cmap_name: str, n_samples: int = 256) -> list:
    """Converts a Matplotlib colormap into a Plotly-compatible colorscale."""
    cmap = plt.get_cmap(cmap_name)
    colorscale = []

    for i in range(n_samples):
        val = i / (n_samples - 1)
        r, g, b, _ = [int(255 * c) for c in cmap(val)]
        colorscale.append([val, f"rgb({r},{g},{b})"])

    return colorscale

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

st.sidebar.header("Display")
hillshade_on = st.sidebar.checkbox("Hillshade", value=True)
state_lines_on = st.sidebar.checkbox("State boundaries", value=True)
county_lines_on = st.sidebar.checkbox("County boundaries", value=False)
lake_lines_on = st.sidebar.checkbox("Lakes & inland water", value=False)
colormap_choice = st.sidebar.selectbox("Colormap", list(COLORSCALES))

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read_basemap_window(bbox, basemap_source):
    return fetch_data.read_basemap_window(bbox, source=basemap_source)


@st.cache_data(show_spinner=False)
def _fetch_and_load(bbox, demtype, api_key):
    path = fetch_data.fetch_dem(bbox, demtype=demtype, api_key=api_key or None)
    return fetch_data.load_elevation(path)


if source_mode == "Local basemap (offline)":
    state_key = ("basemap", basemap_source, bbox)
    need_load = "data" not in st.session_state or st.session_state.get("bbox_key") != state_key
    if need_load:
        with st.spinner("Reading area from local basemap..."):
            try:
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


def _add_contour_lines(fig, display_data, bbox, interval=100):
    west, south, east, north = bbox

    # 1. Mask missing or non-finite values
    valid_data = display_data[np.isfinite(display_data)]
    if valid_data.size == 0:
        return

    min_elev = np.floor(valid_data.min() / interval) * interval
    max_elev = np.ceil(valid_data.max() / interval) * interval
    levels = np.arange(min_elev, max_elev + interval, interval)

    if len(levels) == 0:
        return

    # 2. Build coordinate grids for contour calculation
    rows, cols = display_data.shape
    x_coords = np.linspace(west, east, cols)
    y_coords = np.linspace(north, south, rows)  # North to south

    # 3. Calculate contour paths headlessly using Matplotlib
    fig_temp, ax_temp = plt.subplots()
    cs = ax_temp.contour(x_coords, y_coords, display_data, levels=levels)
    plt.close(fig_temp)

    # 4. Parse path segments using path codes to handle MOVETO / CLOSEPOLY properly
    all_x, all_y = [], []

    for path in cs.get_paths():
        # iter_segments returns (vertices, code)
        for vertices, code in path.iter_segments():
            # MOVETO means a new disconnected segment starts -> lift pen
            if code == Path.MOVETO:
                if all_x and all_x[-1] is not None:
                    all_x.append(None)
                    all_y.append(None)
                all_x.append(vertices[0])
                all_y.append(vertices[1])

            # LINETO or CURVE4 -> draw to point
            elif code in (Path.LINETO, Path.CURVE3, Path.CURVE4):
                all_x.append(vertices[0])
                all_y.append(vertices[1])

            # CLOSEPOLY -> close loop back to start of sub-path
            elif code == Path.CLOSEPOLY:
                if all_x and all_x[-1] is not None:
                    all_x.append(None)
                    all_y.append(None)

        # Break sub-paths when moving to the next path object
        if all_x and all_x[-1] is not None:
            all_x.append(None)
            all_y.append(None)

    # 5. Add contour trace to Plotly figure
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

display_data = downsample_for_display(data)
if display_data.shape != data.shape:
    st.sidebar.caption(
        f"Displaying downsampled to {display_data.shape[1]}×{display_data.shape[0]} px "
        "(full-res data is used for elevation calculations)."
    )

from scipy.ndimage import gaussian_filter

if hillshade_on:
    # 1. Fill NaNs with baseline elevation
    filled = np.where(np.isfinite(display_data), display_data, vmin)
    
    # 2. Smooth terrain slightly to remove pixelation/aliasing noise (sigma=0.8 to 1.2)
    smoothed_dem = gaussian_filter(filled, sigma=0.8)

    # 3. Use realistic sun angle (45° altitude, 315° NW azimuth)
    ls = LightSource(azdeg=315, altdeg=45)
    
    cmap = plt.get_cmap(colorscale)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # 4. Use "soft" blending for balanced contrast
    rgb = ls.shade(
        smoothed_dem, 
        cmap=cmap, 
        norm=norm, 
        blend_mode="soft", 
        vert_exag=0.5  # Subtle vertical exaggeration
    )
    
    rgb_img = (np.clip(rgb[..., :3], 0, 1) * 255).astype(np.uint8)

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

if state_lines_on:
    try:
        _add_state_boundaries(fig, (west, south, east, north))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not load state boundaries: {exc}")

if county_lines_on:
    try:
        _add_county_boundaries(fig, (west, south, east, north))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not load county boundaries: {exc}")
if lake_lines_on:
    try:
        _add_lake_boundaries(fig, (west, south, east, north))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not load lake boundaries: {exc}")

if contour_on:
    try:
        _add_contour_lines(
            fig, 
            display_data, 
            (west, south, east, north), 
            interval=contour_interval
        )
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