"""Fetch fast-growing repositories and their fork histories from the GitHub API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

API = "https://api.github.com"
PER_PAGE = 100
# Stargazer lists are only visible to a repository's owner, so growth over time is
# reconstructed from the fork list, which is public and carries a timestamp per fork.
# Pagination is capped here to keep one repo from using the whole rate limit.
FORK_LIMIT = 40_000


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    description: str
    language: str
    topics: tuple[str, ...]
    stars: int
    forks: int
    created_at: datetime

    def age_days(self, now: datetime) -> float:
        return max((now - self.created_at).total_seconds() / 86400, 1.0)

    def velocity(self, now: datetime) -> float:
        """Stars gained per day since the repository was created."""
        return self.stars / self.age_days(now)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_repo(item: dict) -> Repo:
    return Repo(
        name=item["full_name"],
        url=item["html_url"],
        description=item.get("description") or "",
        language=item.get("language") or "Unknown",
        topics=tuple(item.get("topics") or ()),
        stars=item["stargazers_count"],
        forks=item["forks_count"],
        created_at=parse_time(item["created_at"]),
    )


def _get(path: str, params: dict, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "trend-radar"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{API}{path}?{urllib.parse.urlencode(params)}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_repos(days: int = 30, limit: int = 100, min_stars: int = 50,
                now: datetime | None = None) -> list[Repo]:
    """Return up to `limit` repos created in the last `days` days, most starred first."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).date().isoformat()

    repos: list[Repo] = []
    page = 1
    while len(repos) < limit:
        items = _get("/search/repositories", {
            "q": f"created:>={since} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": min(PER_PAGE, limit - len(repos)),
            "page": page,
        })["items"]
        if not items:
            break
        repos.extend(parse_repo(item) for item in items)
        page += 1
    return repos[:limit]


def sample_pages(forks: int, samples: int) -> list[int]:
    """Pick up to `samples` fork-list pages, evenly spaced from the first to the last reachable."""
    last = max(1, -(-min(forks, FORK_LIMIT) // PER_PAGE))
    if last <= samples:
        return list(range(1, last + 1))
    step = (last - 1) / (samples - 1)
    return sorted({round(1 + i * step) for i in range(samples)})


def page_points(page: int, entries: list[dict]) -> list[tuple[datetime, int]]:
    """Turn one page of forks into (time, cumulative fork count) points.

    The list is ordered oldest first, so entry `i` of page `p` is fork number (p-1)*100 + i + 1.
    """
    base = (page - 1) * PER_PAGE
    return [(parse_time(entry["created_at"]), base + i + 1) for i, entry in enumerate(entries)]


def fetch_fork_history(repo: Repo, samples: int = 12) -> list[tuple[datetime, int]]:
    """Sample the fork list to reconstruct when the repo's forks arrived."""
    owner_repo = repo.name
    points: list[tuple[datetime, int]] = []
    for page in sample_pages(repo.forks, samples):
        entries = _get(f"/repos/{owner_repo}/forks",
                       {"sort": "oldest", "per_page": PER_PAGE, "page": page})
        points.extend(page_points(page, entries))
    return points


def fetch_histories(repos: list[Repo], samples: int = 12,
                    workers: int = 8) -> dict[str, list[tuple[datetime, int]]]:
    """Fetch fork histories concurrently. Repos whose history cannot be read are left out."""
    def one(repo: Repo):
        try:
            return repo.name, fetch_fork_history(repo, samples)
        except (urllib.error.URLError, KeyError, ValueError):
            return repo.name, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {name: points for name, points in pool.map(one, repos) if points}
