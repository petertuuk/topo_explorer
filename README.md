# Topo Explorer

An interactive topographic map of the USA (and the rest of the world), built
as a Streamlit app. Pick a region or a custom bounding box, and drag the
elevation range slider to stretch the colormap so subtle terrain features
pop out.

## How it works

- **Elevation data** comes from one of two places:
  - **Local basemap (offline):** NOAA ETOPO 2022 global relief (30 or 60
    arc-second), downloaded once to `data/`. Reads are decimated on the fly
    to the display size, and the file gets internal overviews (pre-shrunk
    copies) so even a whole-CONUS view loads in well under a second.
  - **Online high-res:** USGS 3DEP for US areas (no API key, no rate limit)
    or OpenTopography's global datasets (SRTM / Copernicus) elsewhere.
- **Requests are sized to what's displayed.** USGS 3DEP resamples
  server-side, so the app asks for roughly the display resolution (~2400 px
  on the long side): a whole-state view is one or two quick requests, and
  zooming in (a smaller bbox) gets you finer detail, down to 3DEP's native
  ~10 m. For OpenTopography (always native resolution), **Auto** picks the
  coarsest dataset that's still detailed enough for the area.
  Tiles are downloaded in parallel, retried on transient server errors,
  mosaicked with `rasterio.merge`, and cached as GeoTIFFs in `data/`.
- The **elevation range slider** controls the colormap's min/max.
- **Hillshade** blends in simulated sun-angle shading, which makes ridgelines
  and valleys much easier to read than flat color alone.
- **Overlays:** state and county boundaries, lakes, major highways, the top
  10 cities in view, and contour lines at a chosen interval. Vector layers
  are downloaded once and cached in `data/vector/` as GeoParquet.
- The map is a Plotly figure: scroll to zoom, drag a box to select an area,
  hover for elevation. The terrain is sent to the browser as a JPEG with a
  small invisible elevation grid on top for the hover readout and colorbar,
  which keeps every rerun light.

## Setup

```bash
pip install -r requirements.txt
```

For areas outside the US you'll want a free OpenTopography API key:

1. Create an account at https://portal.opentopography.org
2. Go to "myOpenTopo" → "Request API key"
3. Put it in a file named `OT_API_KEY` in the project folder (git-ignored),
   or set it as an environment variable:

```bash
export OPENTOPOGRAPHY_API_KEY=your_key_here
```

(Without a key the app falls back to OpenTopography's shared public demo
key, which is heavily rate-limited. USGS 3DEP needs no key.)

## Usage

```bash
streamlit run app.py
```

- Pick a **preset region** (the 50 states plus a few named areas), or check
  **"Use custom bounding box instead"**, type west/south/east/north, and click
  **Fetch this area**. You can also draw a box on the map with the Box Select
  tool and click **🎯 Read Box Selection** to copy it into the fields.
- Choose the **data source**. For the local basemap, the first time you pick
  one you'll be offered a one-time download (and, for basemaps downloaded by
  older versions of the app, a one-time **Build overviews** step).
- For online data, the **📋 What will be requested** panel shows the dataset,
  effective resolution and number of requests before anything is fetched.
- Drag the **elevation range slider** to adjust contrast — e.g. narrow it to
  0–500 m to bring out coastal-plain detail, or 2000–4000 m to pick apart
  alpine terrain.
- The **⏱️ Timings** panel at the bottom of the sidebar shows where time went
  on the last rerun.

## Project layout

```
topo_explorer/
├── app.py               # Streamlit app (run this)
├── src/
│   ├── fetch_data.py    # USGS 3DEP / OpenTopography clients, request planning,
│   │                    # tiling, caching, local basemap reads
│   └── regions.py       # Named bounding-box presets
├── data/                # Cached GeoTIFFs, basemaps, vector layers (git-ignored)
└── requirements.txt
```

## Notes & limits

- OpenTopography caps request area per dataset and is rate-limited
  (500 calls/24h for academic accounts, 100/24h otherwise); large areas are
  split into tiles automatically.
- Contour lines over very large areas at small intervals produce a lot of
  geometry and will slow the map down — use a larger interval when zoomed out.
- Everything downloaded lives in `data/`. Delete files there to force a
  re-download.
