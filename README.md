# Topo Explorer

An interactive topographic map of the USA, built as a Streamlit app. Pick a
region or a custom bounding box, and drag a proper native range slider to
set the min/max elevation of the colormap so subtle terrain features pop
out.

## How it works

- Elevation data comes from [OpenTopography](https://opentopography.org)'s
  free Global Datasets API (SRTM / Copernicus DEMs), fetched on demand for
  whatever bounding box you select.
- `fetch_data.py` automatically splits large bounding boxes into a grid of
  tiles (each dataset has an area cap per request), mosaics them back
  together with `rasterio.merge`, and caches everything to disk in `data/`
  so revisiting an area is instant.
- The **elevation range slider** in the sidebar is a native Streamlit
  `st.slider` (two-handed, drag or click-and-type) — it controls the
  colormap's min/max, independent of which data is loaded.
- A **hillshade** toggle blends in simulated sun-angle shading on top of the
  elevation colors, which makes ridgelines and valleys much easier to read
  than flat color alone.
- The map itself is a Plotly figure, so scroll-to-zoom and click-drag-to-pan
  work out of the box.

## Setup

```bash
pip install -r requirements.txt
```

You'll need a free OpenTopography API key:

1. Create an account at https://portal.opentopography.org
2. Go to "myOpenTopo" → "Request API key"
3. Either paste it into the app's sidebar field each time, or set it as an
   environment variable so it's pre-filled automatically:

```bash
export OPENTOPOGRAPHY_API_KEY=your_key_here
```

(If you skip this, the app falls back to OpenTopography's public demo key,
which is shared by everyone and rate-limited to ~25 requests total — fine
for a quick look, not for real use.)

## Usage

```bash
streamlit run app.py
```

This opens the app in your browser. From there:

- Pick a **preset region** from the sidebar dropdown (Rockies, Sierra
  Nevada, Grand Canyon, Appalachians, etc.), or check **"Use custom
  bounding box"** and type in west/south/east/north coordinates.
- Pick a **resolution/dataset**, or leave it on **Auto** to let the app pick
  the finest resolution that fits the area's request-size limit.
- Click **"Fetch this area"** to download it (cached after the first time).
- Drag the **elevation range slider** to adjust colormap contrast — e.g.
  narrow it to 0-500 m to bring out coastal-plain detail, or 2000-4000 m to
  pick apart alpine terrain that would otherwise all look the same color.
- Toggle **Hillshade** for relief shading.
- Scroll to zoom and drag to pan on the map itself. To see a specific area
  at higher resolution, narrow the custom bounding box to that area and
  click **Fetch this area** again.

## Project layout

```
topo_explorer/
├── app.py               # Streamlit app (run this)
├── src/
│   ├── fetch_data.py    # OpenTopography API client, tiling, caching
│   └── regions.py       # Named bounding-box presets
├── data/                # Cached downloaded GeoTIFFs (created automatically)
└── requirements.txt
```

## Notes & limits

- OpenTopography caps how much area you can request at once depending on
  resolution (30 m requests are capped tighter than 90 m, which is capped
  tighter than the 15 arc-second dataset). `fetch_data.py` handles this
  automatically by splitting large requests into tiles — just be aware that
  a high-res fetch over a large area can mean several API calls and a
  slower fetch.
- The free API key is rate-limited (500 calls/24h for academic accounts,
  100/24h otherwise), which is generous for interactive exploring.
- Downloaded tiles are cached in `data/` by bounding box + dataset. Delete
  files there if you ever want to force a re-download.
