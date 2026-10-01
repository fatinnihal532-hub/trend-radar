"""Write the Markdown report and the velocity chart."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .analyze import Theme, language_share
from .fetch import Repo

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BAR = "#2a78d6"


def write_chart(repos: list[Repo], now: datetime, path: Path, top: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(repos, key=lambda r: r.velocity(now), reverse=True)[:top][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(ranked) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([r.name for r in ranked], [r.velocity(now) for r in ranked], color=BAR, height=0.55)

    ax.set_title("Fastest-growing new repositories", loc="left", color=INK,
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Stars gained per day since creation", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=INK_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _cell(text: str, width: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def render_markdown(repos: list[Repo], themes: list[Theme], now: datetime, days: int,
                    chart: str | None = None, top: int = 15) -> str:
    lines = [
        f"# GitHub trend radar — {now:%Y-%m-%d}",
        "",
        f"{len(repos)} repositories created in the last {days} days, ranked by stars per day.",
        "",
    ]
    if chart:
        lines += [f"![Fastest-growing new repositories]({chart})", ""]

    lines += ["## Fastest growing", "", "| Repository | Language | Stars | Stars/day | About |",
              "|---|---|--:|--:|---|"]
    for r in sorted(repos, key=lambda r: r.velocity(now), reverse=True)[:top]:
        lines.append(f"| [{r.name}]({r.url}) | {r.language} | {r.stars:,} | "
                     f"{r.velocity(now):,.0f} | {_cell(r.description)} |")

    lines += ["", "## Themes", ""]
    for theme in themes:
        leaders = sorted(theme.repos, key=lambda r: r.stars, reverse=True)[:3]
        names = ", ".join(f"[{r.name}]({r.url})" for r in leaders)
        lines.append(f"- **{theme.label}** — {len(theme.repos)} repos, {theme.stars:,} stars. "
                     f"Leaders: {names}")

    lines += ["", "## Languages", "", "| Language | Repos |", "|---|--:|"]
    lines += [f"| {language} | {count} |" for language, count in language_share(repos)[:10]]
    return "\n".join(lines) + "\n"
