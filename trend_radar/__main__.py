"""Command line interface: python -m trend_radar"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from .analyze import find_themes
from .fetch import fetch_histories, fetch_repos
from .growth import backtest, build_curve
from .report import (by_velocity, render_markdown, theme_series, write_curves_chart,
                     write_themes_chart, write_velocity_chart)

# Without a token GitHub allows 60 requests an hour, which covers about five fork histories.
HISTORY_WITHOUT_TOKEN = 5
# Below this many forks the curve is too coarse to say anything about its shape.
MIN_FORKS = 20


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trend_radar",
                                     description="Report what is taking off on GitHub right now.")
    parser.add_argument("--days", type=int, default=30, help="look at repos created in the last N days")
    parser.add_argument("--limit", type=int, default=100, help="how many repos to analyse")
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--themes", type=int, default=6, help="number of themes to cluster into")
    parser.add_argument("--history", type=int, default=None,
                        help="reconstruct fork histories for the N fastest repos (default: all "
                             f"with GITHUB_TOKEN set, {HISTORY_WITHOUT_TOKEN} without)")
    parser.add_argument("--samples", type=int, default=12, help="fork-list pages sampled per repo")
    parser.add_argument("--out", default="report", help="output directory")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    try:
        repos = fetch_repos(days=args.days, limit=args.limit, min_stars=args.min_stars, now=now)
    except urllib.error.HTTPError as e:
        hint = " Set GITHUB_TOKEN to raise the rate limit." if e.code in (403, 429) else ""
        print(f"GitHub API returned {e.code}.{hint}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Could not reach GitHub: {e.reason}", file=sys.stderr)
        return 1
    if not repos:
        print("No repositories matched.", file=sys.stderr)
        return 1

    history = args.history
    if history is None:
        history = len(repos) if os.environ.get("GITHUB_TOKEN") else HISTORY_WITHOUT_TOKEN
    tracked = [r for r in by_velocity(repos, now)[:history] if r.forks >= MIN_FORKS]
    by_name = {r.name: r for r in tracked}
    curves = {name: build_curve(by_name[name], points, now)
              for name, points in fetch_histories(tracked, samples=args.samples).items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    themes = find_themes(repos, k=args.themes)

    charts = {"velocity": "velocity.png"}
    write_velocity_chart(repos, now, out / "velocity.png")
    usable = [c for c in curves.values() if not c.truncated]
    if usable:
        write_curves_chart(repos, curves, now, out / "curves.png")
        charts["curves"] = "curves.png"
    series = theme_series(themes, curves, now, args.days)
    if series:
        write_themes_chart(series, out / "themes.png")
        charts["themes"] = "themes.png"

    report = out / "report.md"
    report.write_text(
        render_markdown(repos, themes, now, args.days, curves=curves,
                        backtest=backtest(usable) if usable else None, charts=charts),
        encoding="utf-8",
    )
    print(f"Analysed {len(repos)} repos ({len(curves)} with fork history) "
          f"in {len(themes)} themes -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
