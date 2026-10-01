"""Group repositories into themes with TF-IDF and k-means."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .fetch import Repo

_TOKEN = re.compile(r"[a-z][a-z0-9+#.-]*[a-z0-9+#]|[a-z]")
_STOPWORDS = frozenset(
    "a an and are as at be best build built by can for from get has have how in into is it "
    "its just like make more most my new no not of on one open or other our out simple so "
    "source than that the their this to up use used using via was we what when which who "
    "will with without you your all any app based code easy fast free github project repo "
    "repository support tool tools written".split()
)


@dataclass(frozen=True)
class Theme:
    label: str
    repos: tuple[Repo, ...]

    @property
    def stars(self) -> int:
        return sum(r.stars for r in self.repos)


def tokenize(repo: Repo) -> list[str]:
    words = [w for w in _TOKEN.findall(repo.description.lower()) if w not in _STOPWORDS and len(w) > 2]
    # Topics are curated by the author, so they count double.
    return words + list(repo.topics) * 2


def tfidf(docs: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Return an L2-normalised TF-IDF matrix and its vocabulary."""
    doc_freq = Counter(term for doc in docs for term in set(doc))
    vocab = sorted(term for term, count in doc_freq.items() if count >= 2)
    index = {term: i for i, term in enumerate(vocab)}
    matrix = np.zeros((len(docs), len(vocab)))
    for row, doc in enumerate(docs):
        for term, count in Counter(doc).items():
            if term in index:
                matrix[row, index[term]] = count * math.log(len(docs) / doc_freq[term])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms), vocab


def kmeans(matrix: np.ndarray, k: int, seed: int = 0, iterations: int = 50) -> np.ndarray:
    """Spherical k-means with k-means++ seeding. Returns a cluster index per row."""
    rng = np.random.default_rng(seed)
    centers = [matrix[rng.integers(len(matrix))]]
    while len(centers) < k:
        distance = 1 - np.max(matrix @ np.array(centers).T, axis=1)
        distance = np.clip(distance, 0, None) ** 2
        if distance.sum() == 0:
            break
        centers.append(matrix[rng.choice(len(matrix), p=distance / distance.sum())])
    centers = np.array(centers)

    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(iterations):
        new_labels = np.argmax(matrix @ centers.T, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(len(centers)):
            members = matrix[labels == c]
            if len(members):
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                centers[c] = mean / norm if norm else mean
    return labels


def find_themes(repos: list[Repo], k: int = 6, seed: int = 0) -> list[Theme]:
    """Cluster repos into at most `k` themes, largest total star count first."""
    if not repos:
        return []
    docs = [tokenize(r) for r in repos]
    matrix, vocab = tfidf(docs)
    if not vocab:
        return [Theme("uncategorised", tuple(repos))]

    labels = kmeans(matrix, min(k, len(repos)), seed=seed)
    themes = []
    for cluster in sorted(set(labels.tolist())):
        members = [r for r, label in zip(repos, labels) if label == cluster]
        weights = matrix[labels == cluster].sum(axis=0)
        top = [vocab[i] for i in np.argsort(weights)[::-1][:3] if weights[i] > 0]
        themes.append(Theme(" / ".join(top) or "uncategorised", tuple(members)))
    return sorted(themes, key=lambda t: t.stars, reverse=True)


def language_share(repos: list[Repo]) -> list[tuple[str, int]]:
    return Counter(r.language for r in repos).most_common()
