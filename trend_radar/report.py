"""Write the Markdown report and its charts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .analyze import Theme, language_share
from .fetch import Repo
from .growth import Curve, daily_gains, forecast, momentum

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FORECAST_DAYS = 7


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    getattr(ax, f"{grid_axis}axis").grid(True, color=INK_MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def by_velocity(repos: list[Repo], now: datetime) -> list[Repo]:
    return sorted(repos, key=lambda r: r.velocity(now), reverse=True)


def write_velocity_chart(repos: list[Repo], now: datetime, path: Path, top: int = 15) -> None:
    plt = _pyplot()
    ranked = by_velocity(repos, now)[:top][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(ranked) + 1.4), facecolor=SURFACE)
    ax.barh([r.name for r in ranked], [r.velocity(now) for r in ranked], color=SERIES[0], height=0.55)
    ax.set_title("Fastest-growing new repositories", loc="left", color=INK,
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Stars gained per day since creation", color=INK_SECONDARY, fontsize=9)
    _style(ax)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def write_curves_chart(repos: list[Repo], curves: dict[str, Curve], now: datetime, path: Path,
                       top: int = 12, columns: int = 4) -> None:
    """Small multiples: how each repo reached its current fork count."""
    plt = _pyplot()
    ranked = [r for r in by_velocity(repos, now) if r.name in curves and not curves[r.name].truncated][:top]
    if not ranked:
        return
    rows = -(-len(ranked) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.9 * columns, 2.1 * rows + 0.7),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, repo in zip(axes.flat, ranked):
        curve = curves[repo.name]
        ax.plot(curve.days, 100 * curve.forks / curve.forks[-1], color=SERIES[0], linewidth=2)
        ax.set_title(repo.name.split("/")[1][:26], loc="left", color=INK, fontsize=9, fontweight="bold")
        ax.text(0.98, 0.06, momentum(curve).shape, transform=ax.transAxes, ha="right",
                color=INK_SECONDARY, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0", "50%", "100%"])
        _style(ax, "y")
    for ax in axes.flat[len(ranked):]:
        ax.set_visible(False)
    fig.suptitle("How they got here: share of today's forks, by days since creation",
                 x=0.012, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def theme_series(themes: list[Theme], curves: dict[str, Curve], now: datetime,
                 days: int) -> list[tuple[Theme, np.ndarray]]:
    """Forks gained per day by each theme over the last `days` days."""
    series = []
    for theme in themes:
        total = np.zeros(days)
        for repo in theme.repos:
            curve = curves.get(repo.name)
            if curve is not None and not curve.truncated:
                total += daily_gains(repo, curve, now, days)
        if total.any():
            series.append((theme, total))
    return series


def write_themes_chart(series: list[tuple[Theme, np.ndarray]], path: Path) -> None:
    plt = _pyplot()
    series = series[:len(SERIES)]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for color, (theme, values) in zip(SERIES, series):
        ax.plot(np.arange(-len(values) + 1, 1), values, color=color, linewidth=2, label=theme.label)
    ax.set_title("Forks gained per day, by theme", loc="left", color=INK,
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Days before this report", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(bottom=0)
    _style(ax, "y")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _cell(text: str, width: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def render_markdown(repos: list[Repo], themes: list[Theme], now: datetime, days: int,
                    curves: dict[str, Curve] | None = None, backtest: dict[str, float] | None = None,
                    charts: dict[str, str] | None = None, top: int = 15) -> str:
    curves = curves or {}
    charts = charts or {}
    lines = [
        f"# GitHub trend radar — {now:%Y-%m-%d}",
        "",
        f"{len(repos)} repositories created in the last {days} days, ranked by stars per day.",
        "",
    ]
    if "velocity" in charts:
        lines += [f"![Fastest-growing new repositories]({charts['velocity']})", ""]

    lines += ["## Fastest growing", "",
              f"| Repository | Language | Stars | Stars/day | Forks | Now vs peak | Shape | Forks in {FORECAST_DAYS}d | About |",
              "|---|---|--:|--:|--:|--:|---|--:|---|"]
    for r in by_velocity(repos, now)[:top]:
        curve = curves.get(r.name)
        if curve is None or curve.truncated:
            growth = "– | – | –"
        else:
            m = momentum(curve)
            growth = f"{m.ratio:.0%} | {m.shape} | {forecast(curve, FORECAST_DAYS):,.0f}"
        lines.append(f"| [{r.name}]({r.url}) | {r.language} | {r.stars:,} | {r.velocity(now):,.0f} | "
                     f"{r.forks:,} | {growth} | {_cell(r.description)} |")
    if curves:
        lines += ["", "GitHub only shows a repository's stargazer list to its owner, so growth over time is "
                      "measured from forks, whose timestamps are public. *Now vs peak* compares forks "
                      "per day in the most recent window with the repo's best window ever. A dash "
                      "means the repo has too few forks to draw a curve."]

    if "curves" in charts:
        lines += ["", "## Growth curves", "", f"![Fork growth curves]({charts['curves']})"]

    if backtest:
        lines += ["", "## How good is the forecast?", "",
                  "Each curve's last 30% was hidden, predicted from the first 70%, and compared "
                  "with what really happened. Median error in total forks:", "",
                  "| Model | Median error |", "|---|--:|"]
        names = {"flat": "Flat (no more forks)", "linear": "Linear (recent rate continues)",
                 "decay": "Decay (recent rate keeps fading)"}
        lines += [f"| {names[model]} | {error:.1%} |" for model, error in backtest.items()]

    lines += ["", "## Themes", ""]
    if "themes" in charts:
        lines += [f"![Forks per day by theme]({charts['themes']})", ""]
    for theme in themes:
        leaders = sorted(theme.repos, key=lambda r: r.stars, reverse=True)[:3]
        names = ", ".join(f"[{r.name}]({r.url})" for r in leaders)
        lines.append(f"- **{theme.label}** — {len(theme.repos)} repos, {theme.stars:,} stars. "
                     f"Leaders: {names}")

    lines += ["", "## Languages", "", "| Language | Repos |", "|---|--:|"]
    lines += [f"| {language} | {count} |" for language, count in language_share(repos)[:10]]
    return "\n".join(lines) + "\n"
