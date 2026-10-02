# trend-radar

[![tests](https://github.com/fatinnihal532-hub/trend-radar/actions/workflows/tests.yml/badge.svg)](https://github.com/fatinnihal532-hub/trend-radar/actions/workflows/tests.yml)

Find out what is taking off on GitHub right now, how each project got there, and whether
it is still growing.

`trend-radar` does four things:

1. **Ranks** the most-starred repositories created in the last N days by stars per day.
2. **Reconstructs each repo's growth curve** from public fork timestamps, by sampling a few
   pages of the fork list instead of downloading all of it.
3. **Measures momentum and forecasts** the next 7 days with a fitted decay model, and
   **back-tests** that forecast against two simpler baselines.
4. **Clusters** the repos into themes with TF-IDF and k-means, and charts how fast each theme
   is growing day by day.

![Fork growth curves](report/curves.png)

![Forks gained per day by theme](report/themes.png)

A full sample report is in [report/report.md](report/report.md).

## Result of the back-test

On the sample run (97 repositories, 1 October 2026), hiding the last 30% of every curve and
predicting it from the first 70% gave these median errors in total forks:

| Model | Median error |
|---|--:|
| Flat (no more forks) | 11.8% |
| Linear (recent rate continues) | 10.4% |
| Decay (recent rate keeps fading) | **5.1%** |

The numbers change with every run, because the set of trending repositories changes. They are
recomputed and printed in each report.

## Run

```bash
pip install -r requirements.txt
```

```bash
python -m trend_radar --days 30 --limit 100 --out report
```

This writes `report/report.md` and three charts.

Growth curves need about 12 API requests per repository. GitHub allows 60 requests an hour
without a token and 5,000 with one, so set `GITHUB_TOKEN` to analyse every repository; without
it only the five fastest get a curve. Any token works, since only public data is read.

| Option | Default | Meaning |
|---|---|---|
| `--days` | 30 | Only repos created in the last N days |
| `--limit` | 100 | How many repos to analyse |
| `--min-stars` | 50 | Ignore repos below this star count |
| `--themes` | 6 | Maximum number of themes (empty clusters are dropped) |
| `--history` | all with a token, 5 without | How many repos get a growth curve |
| `--samples` | 12 | Fork-list pages sampled per repo |
| `--out` | `report` | Output directory |

## How it works

### Why forks and not stars

GitHub only shows a repository's stargazer list to its owner, so nobody else can see *when*
its stars arrived. Forks are public and each one carries a creation time, so the fork list is
used as the growth signal.

### Sampling the curve

The fork list is sorted oldest first, 100 per page. Entry `i` on page `p` is therefore fork
number `(p − 1) × 100 + i + 1`, and its timestamp says when the repo reached that count. Twelve
evenly spaced pages pin the whole curve down, however many forks there are. The points are
interpolated onto an even 200-point grid ([trend_radar/growth.py](trend_radar/growth.py)).

### Momentum

Forks per day are measured over a sliding window (a quarter of the repo's life, between one
day and a week). *Now vs peak* is the latest window divided by the best window ever:

| Now vs peak | Label |
|---|---|
| 75% or more | still climbing |
| 35% to 75% | steady |
| below 35% | cooling off |

### Forecast

After a launch spike, interest usually fades roughly exponentially. The model fits

```
rate(t) = rate_peak · exp(−(t − t_peak) / τ)
```

to the windowed rates after the peak (a straight-line fit to the log of the rate), then
integrates the current rate forward:

```
forks(T + h) = forks(T) + rate(T) · τ · (1 − exp(−h / τ))
```

If the rate is not falling, τ is infinite and the forecast is a straight line.

### Themes

Each repo becomes a TF-IDF vector over its description and topics (topics count double).
Spherical k-means with k-means++ seeding groups them, and each theme is labelled with its
three heaviest terms. The seed is fixed, so the same input gives the same themes. The
clustering is written directly in NumPy ([trend_radar/analyze.py](trend_radar/analyze.py)).

## Limits

- Forks are a proxy. A repo people star but rarely fork gets a noisy curve, and repos with
  fewer than 20 forks get none.
- Deleted forks vanish from the list, so early counts can be slightly low.
- Forecasts for repos only a day or two old extrapolate from very little data.
- Themes come from keywords, not meaning: descriptions in different languages or with no
  shared vocabulary land in whichever cluster is nearest.

## Tests

```bash
python -m unittest discover -s tests
```

26 tests cover page sampling, curve reconstruction, momentum, the decay fit (it recovers a
known τ from a synthetic curve), the forecast, the back-test and the clustering.
