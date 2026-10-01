"""Command line interface: python -m trend_radar"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from .analyze import find_themes
from .fetch import fetch_repos
from .report import render_markdown, write_chart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trend_radar",
                                     description="Report what is taking off on GitHub right now.")
    parser.add_argument("--days", type=int, default=30, help="look at repos created in the last N days")
    parser.add_argument("--limit", type=int, default=100, help="how many repos to analyse")
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--themes", type=int, default=6, help="number of themes to cluster into")
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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_chart(repos, now, out / "velocity.png")
    themes = find_themes(repos, k=args.themes)
    report = out / "report.md"
    report.write_text(render_markdown(repos, themes, now, args.days, chart="velocity.png"),
                      encoding="utf-8")
    print(f"Analysed {len(repos)} repos in {len(themes)} themes -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
