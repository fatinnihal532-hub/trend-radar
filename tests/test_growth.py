import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from trend_radar.fetch import Repo, page_points, sample_pages
from trend_radar.growth import (GRID, Curve, backtest, build_curve, daily_gains, decay_time,
                                forecast, momentum)

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def repo(forks, age_days):
    return Repo(name="o/r", url="", description="", language="Python", topics=(), stars=forks * 10,
                forks=forks, created_at=NOW - timedelta(days=age_days))


def curve_from(fn, age=20.0):
    days = np.linspace(0.0, age, GRID)
    return Curve(days, fn(days))


def linear(rate=100.0):
    return lambda t: rate * t


def decaying(rate=1000.0, tau=4.0):
    # Rate starts at `rate` and fades with time constant tau; total tends to rate * tau.
    return lambda t: rate * tau * (1 - np.exp(-t / tau))


class SamplingTest(unittest.TestCase):
    def test_small_repo_reads_every_page(self):
        self.assertEqual(sample_pages(250, samples=12), [1, 2, 3])

    def test_large_repo_is_sampled_from_first_to_last_page(self):
        pages = sample_pages(10_000, samples=5)
        self.assertEqual(pages, [1, 26, 50, 75, 100])

    def test_pages_stop_at_the_api_limit(self):
        self.assertEqual(sample_pages(90_000, samples=3)[-1], 400)

    def test_page_points_number_forks_by_position(self):
        entries = [{"created_at": f"2026-09-0{d}T00:00:00Z"} for d in (1, 2, 3)]
        points = page_points(3, entries)
        self.assertEqual([count for _, count in points], [201, 202, 203])
        self.assertEqual(points[0][0], datetime(2026, 9, 1, tzinfo=timezone.utc))

    def test_empty_page(self):
        self.assertEqual(page_points(1, []), [])


class CurveTest(unittest.TestCase):
    def test_curve_starts_at_zero_ends_at_current_forks_and_never_falls(self):
        r = repo(forks=1000, age_days=10)
        points = [(r.created_at + timedelta(days=2), 600), (r.created_at + timedelta(days=1), 100),
                  (r.created_at + timedelta(days=5), 550)]  # out of order and non-monotone
        curve = build_curve(r, points, NOW)

        self.assertEqual(curve.forks[0], 0)
        self.assertEqual(curve.forks[-1], 1000)
        self.assertTrue(np.all(np.diff(curve.forks) >= 0))
        self.assertAlmostEqual(curve.at(1.0), 100, delta=20)

    def test_repo_past_the_fork_limit_is_marked_truncated(self):
        self.assertTrue(build_curve(repo(50_000, 10), [], NOW).truncated)
        self.assertFalse(build_curve(repo(5_000, 10), [], NOW).truncated)


class MomentumTest(unittest.TestCase):
    def test_constant_growth_is_still_climbing(self):
        m = momentum(curve_from(linear(100)))
        self.assertAlmostEqual(m.recent_rate, 100, delta=1)
        self.assertAlmostEqual(m.ratio, 1.0, delta=0.01)
        self.assertEqual(m.shape, "still climbing")

    def test_launch_spike_is_cooling_off(self):
        m = momentum(curve_from(decaying()))
        self.assertLess(m.ratio, 0.1)
        self.assertEqual(m.shape, "cooling off")


class ForecastTest(unittest.TestCase):
    def test_decay_time_is_recovered(self):
        self.assertAlmostEqual(decay_time(curve_from(decaying(tau=4.0))), 4.0, delta=0.2)

    def test_decay_time_is_infinite_when_growth_is_not_slowing(self):
        self.assertEqual(decay_time(curve_from(linear())), float("inf"))

    def test_linear_growth_is_extrapolated_exactly(self):
        curve = curve_from(linear(100), age=20)
        self.assertAlmostEqual(forecast(curve, 5, "linear"), 2500, delta=5)
        self.assertAlmostEqual(forecast(curve, 5, "decay"), 2500, delta=5)
        self.assertEqual(forecast(curve, 5, "flat"), 2000)

    def test_decay_model_beats_linear_on_a_fading_spike(self):
        truth = decaying(rate=1000, tau=4.0)
        seen = curve_from(truth, age=10)
        actual = truth(20.0)

        decay_error = abs(forecast(seen, 10, "decay") - actual)
        linear_error = abs(forecast(seen, 10, "linear") - actual)

        self.assertLess(decay_error, linear_error)
        self.assertLess(decay_error / actual, 0.05)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            forecast(curve_from(linear()), 1, "magic")


class BacktestTest(unittest.TestCase):
    def test_backtest_reports_each_model_and_skips_truncated_curves(self):
        steady = curve_from(linear(100))
        spike = curve_from(decaying())
        days = np.linspace(0, 20, GRID)
        truncated = Curve(days, 3000 * days, truncated=True)

        errors = backtest([steady, spike, truncated])

        self.assertEqual(set(errors), {"flat", "linear", "decay"})
        self.assertLess(errors["decay"], 0.02)
        self.assertGreater(errors["flat"], errors["decay"])

    def test_backtest_with_nothing_usable(self):
        self.assertEqual(backtest([]), {})


class DailyGainsTest(unittest.TestCase):
    def test_gains_are_zero_before_the_repo_existed(self):
        r = repo(forks=1000, age_days=10)
        gains = daily_gains(r, curve_from(linear(100), age=10), NOW, days=30)

        self.assertEqual(len(gains), 30)
        self.assertTrue(np.all(gains[:19] == 0))
        self.assertAlmostEqual(gains[-1], 100, delta=1)
        self.assertAlmostEqual(gains.sum(), 1000, delta=1)


if __name__ == "__main__":
    unittest.main()
