#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_ato_mpl")

import pandas as pd

from rl_ato.plotting import plot_sensitivity_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Render manuscript-style sensitivity figures.")
    parser.add_argument("--summary", default="outputs/sensitivity_summary.csv")
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--out-dir", default="outputs/sensitivity_figures")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--no-pairs", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    episode_path = args.episodes
    if episode_path is None:
        candidate = Path(args.summary).with_name(f"{Path(args.summary).stem}_episodes.csv")
        episode_path = str(candidate) if candidate.exists() else None
    episodes = pd.read_csv(episode_path) if episode_path else None
    formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
    saved = plot_sensitivity_figures(
        summary,
        episodes,
        args.out_dir,
        formats=formats,
        overview=not args.no_overview,
        paired=not args.no_pairs,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
