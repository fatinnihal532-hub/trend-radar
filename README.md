# trend-radar

Find out what is taking off on GitHub right now, and what it has in common.

`trend-radar` pulls the most-starred repositories created in the last N days, ranks them by
**stars per day** (so a 3-day-old repo with 3,000 stars beats a 30-day-old one with 10,000),
and clusters their descriptions and topics into themes with TF-IDF and k-means.

![Fastest-growing new repositories](report/velocity.png)

See a full sample in [report/report.md](report/report.md).

## Run

```bash
pip install -r requirements.txt
```

```bash
python -m trend_radar --days 30 --limit 100 --themes 6 --out report
```

This writes `report/report.md` and `report/velocity.png`.

| Option | Default | Meaning |
|---|---|---|
| `--days` | 30 | Only repos created in the last N days |
| `--limit` | 100 | How many repos to analyse |
| `--min-stars` | 50 | Ignore repos below this star count |
| `--themes` | 6 | Maximum number of themes (empty clusters are dropped) |
| `--out` | `report` | Output directory |

No token is needed. Set `GITHUB_TOKEN` if you hit the unauthenticated search rate limit
(10 requests per minute).

## How it works

1. **Fetch** – one GitHub Search API query: `created:>=<date> stars:>=<min>` sorted by stars.
2. **Velocity** – `stars / max(age in days, 1)`.
3. **Themes** – each repo becomes a TF-IDF vector over its description and topics (topics
   count double). Spherical k-means with k-means++ seeding groups them; each theme is
   labelled with its three heaviest terms. The seed is fixed, so the same input gives the
   same themes.

The clustering is implemented directly in NumPy in [trend_radar/analyze.py](trend_radar/analyze.py).

## Limits

- Themes come from keywords, not meaning: descriptions in different languages or with
  no shared vocabulary land in whichever cluster is nearest.
- The Search API returns at most 1,000 results per query.

## Tests

```bash
python -m unittest discover -s tests
```
