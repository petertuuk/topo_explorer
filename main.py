#!/usr/bin/env python3
"""Launch the interactive topographic map viewer.

Examples:
    python main.py
    python main.py --region rockies
    python main.py --bbox -122.5 46.6 -120.8 47.6
    python main.py --list-regions
"""

from __future__ import annotations

import argparse
import sys

from src import fetch_data, regions
from src.viewer import TopoViewer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--region",
        default="conus",
        help="Named preset region to start at (default: conus). See --list-regions.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Custom starting bounding box in decimal degrees, overrides --region.",
    )
    parser.add_argument(
        "--demtype",
        default=None,
        help="OpenTopography dataset code (SRTMGL1, SRTMGL3, SRTM15Plus, COP30, COP90, ...). "
        "Auto-selected based on bbox size if omitted.",
    )
    parser.add_argument("--api-key", default=None, help="OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY).")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached data and re-download.")
    parser.add_argument("--list-regions", action="store_true", help="List available --region presets and exit.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_regions:
        for name in sorted(regions.REGIONS):
            print(name)
        return 0

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = regions.get_region(args.region)

    print(f"Fetching initial elevation data for bbox={bbox} ...")
    path = fetch_data.fetch_dem(
        bbox,
        demtype=args.demtype,
        api_key=args.api_key,
        force=args.force_refresh,
        progress_callback=print,
    )

    viewer = TopoViewer(path, api_key=args.api_key)
    viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
