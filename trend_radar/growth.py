"""Reconstruct fork-growth curves, measure their momentum, forecast them and back-test the forecast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .fetch import FORK_LIMIT, Repo

GRID = 200
MIN_DECAY_POINTS = 5

STILL_CLIMBING = 0.75
STEADY = 0.35


@dataclass(frozen=True)
class Curve:
    """Cumulative forks on an even grid of days since the repository was created."""
    days: np.ndarray
    forks: np.ndarray
    truncated: bool = False  # the fork list could not be read all the way to the present

    @property
    def age(self) -> float:
        return float(self.days[-1])

    def at(self, day: float) -> float:
        return float(np.interp(day, self.days, self.forks))

    def until(self, day: float) -> "Curve":
        """The curve as it would have looked on `day`, for back-testing."""
        days = np.linspace(0.0, day, GRID)
        return Curve(days, np.interp(days, self.days, self.forks), self.truncated)


def build_curve(repo: Repo, points: list[tuple[datetime, int]], now: datetime) -> Curve:
    age = repo.age_days(now)
    raw = [(0.0, 0.0)]
    raw += [((when - repo.created_at).total_seconds() / 86400, float(count)) for when, count in points]
    raw.append((age, float(repo.forks)))
    raw = sorted((min(max(day, 0.0), age), count) for day, count in raw)

    days = np.array([d for d, _ in raw])
    forks = np.maximum.accumulate(np.array([c for _, c in raw]))
    # np.interp needs strictly increasing x; keep the highest count seen at each instant.
    keep = np.append(np.diff(days) > 0, True)
    grid = np.linspace(0.0, age, GRID)
    return Curve(grid, np.interp(grid, days[keep], forks[keep]), truncated=repo.forks > FORK_LIMIT)


def window_days(age: float) -> float:
    """Width of the 'recent' window: a quarter of the repo's life, between one day and a week."""
    return float(min(7.0, max(1.0, age / 4)))


def window_rates(curve: Curve) -> tuple[np.ndarray, np.ndarray]:
    """Forks per day over a sliding window. Returns (day at window end, rate)."""
    step = curve.days[1] - curve.days[0]
    k = max(2, min(GRID - 1, round(window_days(curve.age) / step)))
    rates = (curve.forks[k:] - curve.forks[:-k]) / (curve.days[k:] - curve.days[:-k])
    return curve.days[k:], rates


@dataclass(frozen=True)
class Momentum:
    recent_rate: float  # forks per day over the most recent window
    peak_rate: float    # the best window the repo ever had
    ratio: float        # recent / peak: 1.0 means it is growing as fast as it ever has

    @property
    def shape(self) -> str:
        if self.ratio >= STILL_CLIMBING:
            return "still climbing"
        if self.ratio >= STEADY:
            return "steady"
        return "cooling off"


def momentum(curve: Curve) -> Momentum:
    _, rates = window_rates(curve)
    peak = float(rates.max())
    recent = float(rates[-1])
    return Momentum(recent, peak, recent / peak if peak > 0 else 0.0)


def decay_time(curve: Curve) -> float:
    """Fit rate(t) = rate_peak * exp(-(t - t_peak) / tau) after the peak and return tau in days.

    Returns infinity when the rate is not falling or there is too little data after the peak.
    """
    days, rates = window_rates(curve)
    peak = int(np.argmax(rates))
    days, rates = days[peak:], rates[peak:]
    positive = rates > 0
    if positive.sum() < MIN_DECAY_POINTS:
        return float("inf")
    slope = np.polyfit(days[positive], np.log(rates[positive]), 1)[0]
    # A fade slower than 100 lifetimes is indistinguishable from no fade at all.
    if slope >= 0 or -1 / slope > 100 * curve.age:
        return float("inf")
    return float(-1 / slope)


def forecast(curve: Curve, horizon: float, model: str = "decay") -> float:
    """Predict total forks `horizon` days after the end of the curve.

    Models: "flat" (no more forks), "linear" (the recent rate continues) and
    "decay" (the recent rate keeps fading at the pace fitted since the peak).
    """
    now = float(curve.forks[-1])
    if model == "flat":
        return now
    rate = momentum(curve).recent_rate
    if model == "linear":
        return now + rate * horizon
    if model == "decay":
        tau = decay_time(curve)
        if not np.isfinite(tau):
            return now + rate * horizon
        return now - rate * tau * np.expm1(-horizon / tau)
    raise ValueError(f"unknown model: {model}")


MODELS = ("flat", "linear", "decay")


def backtest(curves: list[Curve], cut: float = 0.7) -> dict[str, float]:
    """Hide the last part of every curve, forecast it, and return each model's median error.

    The error is |predicted - actual| / actual on the total fork count at the end of the curve.
    Curves that do not reach the present (more than 40,000 forks) are skipped.
    """
    errors: dict[str, list[float]] = {model: [] for model in MODELS}
    for curve in curves:
        if curve.truncated or curve.forks[-1] <= 0:
            continue
        seen = curve.until(curve.age * cut)
        horizon = curve.age - seen.age
        for model in MODELS:
            predicted = forecast(seen, horizon, model)
            errors[model].append(abs(predicted - curve.forks[-1]) / curve.forks[-1])
    return {model: float(np.median(values)) for model, values in errors.items() if values}


def daily_gains(repo: Repo, curve: Curve, now: datetime, days: int) -> np.ndarray:
    """Forks gained on each of the last `days` calendar days (oldest first)."""
    age = repo.age_days(now)
    offsets = np.arange(-days, 1)  # days relative to now
    cumulative = np.interp(age + offsets, curve.days, curve.forks, left=0.0)
    return np.diff(cumulative)
