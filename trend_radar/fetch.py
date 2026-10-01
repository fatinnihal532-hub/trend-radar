"""Fetch recently created, fast-growing repositories from the GitHub Search API."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

API = "https://api.github.com/search/repositories"
PER_PAGE = 100


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

    def velocity(self, now: datetime) -> float:
        """Stars gained per day since the repository was created."""
        age_days = max((now - self.created_at).total_seconds() / 86400, 1.0)
        return self.stars / age_days


def parse_repo(item: dict) -> Repo:
    return Repo(
        name=item["full_name"],
        url=item["html_url"],
        description=item.get("description") or "",
        language=item.get("language") or "Unknown",
        topics=tuple(item.get("topics") or ()),
        stars=item["stargazers_count"],
        forks=item["forks_count"],
        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
    )


def fetch_repos(days: int = 30, limit: int = 100, min_stars: int = 50,
                now: datetime | None = None) -> list[Repo]:
    """Return up to `limit` repos created in the last `days` days, most starred first.

    Set GITHUB_TOKEN to raise the rate limit from 10 to 30 searches per minute.
    """
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).date().isoformat()
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "trend-radar"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos: list[Repo] = []
    page = 1
    while len(repos) < limit:
        query = urllib.parse.urlencode({
            "q": f"created:>={since} stars:>={min_stars}",
            "sort": "stars",
            "order": "desc",
            "per_page": min(PER_PAGE, limit - len(repos)),
            "page": page,
        })
        request = urllib.request.Request(f"{API}?{query}", headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            items = json.load(response)["items"]
        if not items:
            break
        repos.extend(parse_repo(item) for item in items)
        page += 1
    return repos[:limit]
