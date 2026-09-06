"""Topo Explorer - Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LightSource, Normalize
import matplotlib.pyplot as plt
import geopandas as gpd

from src import fetch_data, regions

st.set_page_config(page_title="Topo Explorer", layout="wide")
st.title("🗺️ Topo Explorer")
st.caption("Elevation data from NOAA ETOPO 2022 (local, offline) or OpenTopography (rate-limited API). Drag the slider to stretch the colormap around the terrain you care about.")

DEMTYPE_LABELS = {
    "Auto": None,
    "SRTMGL1 (~30 m)": "SRTMGL1",
    "SRTMGL3 (~90 m)": "SRTMGL3",
    "SRTM15Plus (~450 m)": "SRTM15Plus",
    "COP30 (~30 m)": "COP30",
    "COP90 (~90 m)": "COP90",
}
COLORSCALES = {"Earth tones": "gist_earth", "Viridis": "viridis", "Turbo": "turbo", "Greys": "gray"}

# Streamlit/browser can choke on sending tens of millions of pixels as chart
# data (and no monitor can show that much detail anyway). Cap whichever
# dimension is larger to this many pixels before building the figure; the
# full-resolution array from disk is untouched, this only affects display.
MAX_DISPLAY_DIM = 1400

STATE_BOUNDARIES_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2025/shp/"
    "cb_2025_us_state_500k.zip"
)


# def downsample_for_display(data: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
#     rows, cols = data.shape
#     step = max(1, -(-max(rows, cols) // max_dim))  # ceil division
#     if step == 1:
#         return data
#     return data[::step, ::step]

from scipy.ndimage import gaussian_filter
def downsample_for_display(data: np.ndarray, max_dim: int = MAX_DISPLAY_DIM) -> np.ndarray:
    rows, cols = data.shape
    step = max(1, -(-max(rows, cols) // max_dim))  # ceil division
    if step == 1:
        return data
    
    # Sigma proportional to step size acts as an anti-aliasing low-pass filter
    smoothed = gaussian_filter(data.astype(np.float32), sigma=step / 2.0, mode='nearest')
    return smoothed[::step, ::step]

@st.cache_data(show_spinner=False)
def _load_state_boundaries():
    states = gpd.read_file(STATE_BOUNDARIES_URL)
    return states.to_crs("EPSG:4326")


import plotly.graph_objects as go
from shapely.ops import clip_by_rect

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

        # Standardize handling for Polygons, LineStrings, and Multi-geometries
        parts = clipped.geoms if hasattr(clipped, "geoms") else [clipped]

        for part in parts:
            if part.is_empty:
                continue

            # Get exterior ring if Polygon, otherwise extract direct line points
            boundary = part.exterior if hasattr(part, "exterior") else part
            x, y = boundary.xy

            # Append coordinate arrays; None acts as a pen-up command between shapes
            all_x.extend(list(x) + [None])
            all_y.extend(list(y) + [None])

    # Batch all geometries into one trace for optimal WebGL/Canvas rendering speed
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

# --------------------------------------------------------------------------
# Sidebar: area selection
# --------------------------------------------------------------------------
st.sidebar.header("Area")

region_names = sorted(regions.REGIONS)
region_choice = st.sidebar.selectbox("Preset region", region_names, index=region_names.index("conus"))

use_custom_bbox = st.sidebar.checkbox("Use custom bounding box instead")
if use_custom_bbox:
    default_bbox = st.session_state.get("last_bbox", regions.REGIONS[region_choice])
    c1, c2 = st.sidebar.columns(2)
    west = c1.number_input("West", value=float(default_bbox[0]), format="%.3f")
    south = c2.number_input("South", value=float(default_bbox[1]), format="%.3f")
    east = c1.number_input("East", value=float(default_bbox[2]), format="%.3f")
    north = c2.number_input("North", value=float(default_bbox[3]), format="%.3f")
    bbox = (west, south, east, north)
else:
    bbox = regions.REGIONS[region_choice]

# --------------------------------------------------------------------------
# Sidebar: data source
# --------------------------------------------------------------------------
st.sidebar.header("Data source")
source_mode = st.sidebar.radio(
    "Where to get elevation data",
    ["Local basemap (recommended)", "OpenTopography API (rate-limited)"],
    index=0,
    help=(
        "Local basemap: download NOAA's ETOPO 2022 global relief model once "
        "(no API key, no rate limit), then browse anywhere offline. "
        "OpenTopography API: higher resolution but capped at a small number "
        "of requests per day."
    ),
)

BASEMAP_LABELS = {
    "etopo30s": "etopo30s — ~1.5 GB, ~930 m/px",
    "etopo60s": "etopo60s — ~444 MB, ~1.8 km/px",
}

if source_mode == "Local basemap (recommended)":
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

            with st.spinner("Downloading basemap - this only happens once..."):
                fetch_data.download_basemap(basemap_source, progress_callback=_report_progress)
            st.rerun()
        st.stop()
else:
    demtype_choice = st.sidebar.selectbox("Resolution / dataset", list(DEMTYPE_LABELS))
    demtype = DEMTYPE_LABELS[demtype_choice]

    api_key = st.sidebar.text_input(
        "OpenTopography API key",
        value=os.environ.get("OPENTOPOGRAPHY_API_KEY", ""),
        type="password",
        help="Free key from portal.opentopography.org. Blank falls back to the shared, rate-limited demo key.",
    )

    fetch_clicked = st.sidebar.button("Fetch this area", type="primary", use_container_width=True)

st.sidebar.header("Display")
hillshade_on = st.sidebar.checkbox("Hillshade", value=True)
state_lines_on = st.sidebar.checkbox("State boundaries", value=True)
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


if source_mode == "Local basemap (recommended)":
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
    state_key = ("opentopo", demtype, bbox)
    need_load = (
        "data" not in st.session_state
        or fetch_clicked
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
# Elevation range slider - the whole point of this rewrite
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

st.sidebar.caption(f"Data range: {elev_min:.0f} m to {elev_max:.0f} m ({data.shape[1]}×{data.shape[0]} px)")

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
west, east, south, north = extent
colorscale = COLORSCALES[colormap_choice]

display_data = downsample_for_display(data)
if display_data.shape != data.shape:
    st.sidebar.caption(
        f"Displaying downsampled to {display_data.shape[1]}×{display_data.shape[0]} px "
        "(full-res data is still used for the elevation range above)."
    )

def matplotlib_to_plotly_colorscale(cmap_name: str, n_samples: int = 256) -> list:
    """Converts a Matplotlib colormap into a Plotly-compatible colorscale."""
    cmap = plt.get_cmap(cmap_name)
    colorscale = []
    
    for i in range(n_samples):
        # Normalize index to 0.0 - 1.0 range
        val = i / (n_samples - 1)
        # Extract RGB values (0-255 scale)
        r, g, b, _ = [int(255 * c) for c in cmap(val)]
        colorscale.append([val, f"rgb({r},{g},{b})"])
        
    return colorscale


if hillshade_on:
    ls = LightSource(azdeg=315, altdeg=45)
    filled = np.where(np.isfinite(display_data), display_data, vmin)
    mpl_cmap_name = colorscale #{"gist_earth": "gist_earth", "viridis": "viridis", "turbo": "turbo", "greys": "gray"}[colorscale]
    cmap = plt.get_cmap(mpl_cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb = ls.shade(filled, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=1.0)
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
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

if state_lines_on:
    try:
        _add_state_boundaries(fig, (west, south, east, north))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not load state boundaries: {exc}")

fig.update_layout(
    height=750,
    margin=dict(l=0, r=0, t=10, b=0),
    dragmode="pan",
    xaxis_title="Longitude",
    yaxis_title="Latitude",
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

st.caption(
    "Scroll to zoom, drag to pan (Plotly's built-in controls). "
    "On the local basemap, any region/bbox change reads instantly from disk - no network calls. "
    "Switch to the OpenTopography API in the sidebar only when you want a specific area at "
    "30-90 m resolution instead of the basemap's ~1-2 km."
)
