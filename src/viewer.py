"""Interactive topographic map viewer.

Shows an elevation raster with a draggable min/max colormap range slider,
an optional hillshade overlay, and a button to re-fetch the currently
visible region at higher resolution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib.widgets import Button, CheckButtons, RangeSlider

from . import fetch_data

CMAP_NAME = "terrain"


class TopoViewer:
    def __init__(
        self,
        initial_path: Path,
        api_key: str | None = None,
        cache_dir: str | Path = fetch_data.DEFAULT_CACHE_DIR,
    ):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cmap = plt.get_cmap(CMAP_NAME).copy()
        self.cmap.set_bad(color="#cfe8f3")  # pale blue for nodata / ocean

        self.data, self.extent = fetch_data.load_elevation(initial_path)
        self.hillshade_on = False

        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(bottom=0.22)
        self.fig.canvas.manager.set_window_title("Topo Explorer")

        vmin, vmax = self._data_range()
        self.im = self.ax.imshow(
            self.data,
            cmap=self.cmap,
            extent=self.extent,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_title("Drag the slider to adjust elevation contrast")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Elevation (m)")

        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=9, color="dimgray")

        self._build_slider(vmin, vmax)
        self._build_hillshade_toggle()
        self._build_fetch_button()

    # -- setup helpers ----------------------------------------------------

    def _data_range(self) -> tuple[float, float]:
        finite = self.data[np.isfinite(self.data)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _build_slider(self, vmin: float, vmax: float) -> None:
        ax_slider = self.fig.add_axes([0.15, 0.09, 0.6, 0.03])
        span = max(vmax - vmin, 1.0)
        self.slider = RangeSlider(
            ax_slider,
            "Elevation range (m)",
            valmin=vmin - 0.05 * span,
            valmax=vmax + 0.05 * span,
            valinit=(vmin, vmax),
        )
        self.slider.on_changed(self._on_slider_change)

    def _build_hillshade_toggle(self) -> None:
        ax_check = self.fig.add_axes([0.8, 0.08, 0.18, 0.07])
        self.check = CheckButtons(ax_check, ["Hillshade"], [False])
        self.check.on_clicked(self._on_toggle_hillshade)

    def _build_fetch_button(self) -> None:
        ax_button = self.fig.add_axes([0.15, 0.03, 0.28, 0.045])
        self.fetch_button = Button(ax_button, "Load high-res for view")
        self.fetch_button.on_clicked(self._on_fetch_highres)

    # -- rendering ----------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _render(self) -> None:
        vmin, vmax = self.slider.val
        if self.hillshade_on:
            rgb = self._hillshaded_rgba(vmin, vmax)
            self.im.set_data(rgb)
        else:
            self.im.set_data(self.data)
            self.im.set_clim(vmin, vmax)
        self.fig.canvas.draw_idle()

    def _hillshaded_rgba(self, vmin: float, vmax: float) -> np.ndarray:
        ls = LightSource(azdeg=315, altdeg=45)
        norm = Normalize(vmin=vmin, vmax=vmax)
        filled = np.where(np.isfinite(self.data), self.data, vmin)
        rgb = ls.shade(
            filled,
            cmap=self.cmap,
            norm=norm,
            blend_mode="soft",
            vert_exag=1.0,
        )
        # Preserve nodata masking as transparency.
        alpha = np.where(np.isfinite(self.data), 1.0, 0.0)
        rgba = np.dstack([rgb[..., :3], alpha])
        return rgba

    # -- callbacks ------------------------------------------------------

    def _on_slider_change(self, _val) -> None:
        self._render()

    def _on_toggle_hillshade(self, _label) -> None:
        self.hillshade_on = not self.hillshade_on
        self._render()

    def _on_fetch_highres(self, _event) -> None:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        west, east = sorted(xlim)
        south, north = sorted(ylim)
        bbox = (west, south, east, north)

        demtype = fetch_data.pick_demtype(bbox, prefer_high_res=True)
        self._set_status(f"Fetching {demtype} for current view...")
        try:
            path = fetch_data.fetch_dem(
                bbox,
                demtype=demtype,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
                progress_callback=self._set_status,
            )
            data, extent = fetch_data.load_elevation(path)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self._set_status(f"Fetch failed: {exc}")
            return

        self.data = data
        self.extent = extent
        self.im.set_extent(extent)
        vmin, vmax = self._data_range()
        span = max(vmax - vmin, 1.0)
        self.slider.valmin = vmin - 0.05 * span
        self.slider.valmax = vmax + 0.05 * span
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
        self.slider.set_val((vmin, vmax))
        self.ax.set_xlim(west, east)
        self.ax.set_ylim(south, north)
        self._set_status(f"Loaded {demtype} ({data.shape[1]}x{data.shape[0]} px) for current view.")
        self._render()

    def show(self) -> None:
        plt.show()
