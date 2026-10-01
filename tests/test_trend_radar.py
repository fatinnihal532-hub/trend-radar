import unittest
from datetime import datetime, timedelta, timezone

from trend_radar.analyze import find_themes, language_share
from trend_radar.fetch import Repo, parse_repo
from trend_radar.report import render_markdown

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def repo(name, description, topics=(), stars=100, age_days=10, language="Python"):
    return Repo(name=name, url=f"https://github.com/{name}", description=description,
                language=language, topics=tuple(topics), stars=stars, forks=0,
                created_at=NOW - timedelta(days=age_days))


AGENTS = [
    repo("a/agent-memory", "Memory layer for LLM agents", ["agents", "llm"], stars=900),
    repo("b/agent-runner", "Orchestrate LLM agents at work", ["agents", "llm"], stars=500),
    repo("c/agent-skills", "Skills for coding agents", ["agents", "llm"], stars=300),
]
VOICE = [
    repo("d/voice-clone", "Local voice cloning and speech synthesis", ["voice", "tts"], stars=200),
    repo("e/dubbing", "Video dubbing with speech synthesis", ["voice", "tts"], stars=100),
    repo("f/dictate", "Offline dictation and voice transcription", ["voice", "tts"], stars=50),
]


class ParseTest(unittest.TestCase):
    def test_parse_repo_handles_missing_fields(self):
        parsed = parse_repo({
            "full_name": "x/y", "html_url": "https://github.com/x/y", "description": None,
            "language": None, "stargazers_count": 7, "forks_count": 1,
            "created_at": "2026-09-21T00:00:00Z",
        })
        self.assertEqual(parsed.description, "")
        self.assertEqual(parsed.language, "Unknown")
        self.assertEqual(parsed.topics, ())
        self.assertAlmostEqual(parsed.velocity(NOW), 0.7)

    def test_velocity_never_divides_by_less_than_a_day(self):
        self.assertEqual(repo("x/y", "", stars=40, age_days=0).velocity(NOW), 40)


class ThemeTest(unittest.TestCase):
    def test_separates_two_obvious_themes(self):
        themes = find_themes(AGENTS + VOICE, k=2)

        self.assertEqual(len(themes), 2)
        groups = [{r.name for r in t.repos} for t in themes]
        self.assertIn({r.name for r in AGENTS}, groups)
        self.assertIn({r.name for r in VOICE}, groups)

    def test_themes_are_ordered_by_stars_and_labelled(self):
        themes = find_themes(AGENTS + VOICE, k=2)
        self.assertEqual({r.name for r in themes[0].repos}, {r.name for r in AGENTS})
        self.assertIn("agents", themes[0].label)

    def test_is_deterministic(self):
        first = [t.label for t in find_themes(AGENTS + VOICE, k=2)]
        second = [t.label for t in find_themes(AGENTS + VOICE, k=2)]
        self.assertEqual(first, second)

    def test_empty_input(self):
        self.assertEqual(find_themes([]), [])

    def test_no_shared_vocabulary_falls_back_to_one_theme(self):
        themes = find_themes([repo("x/y", "alpha"), repo("p/q", "beta")], k=2)
        self.assertEqual([t.label for t in themes], ["uncategorised"])


class ReportTest(unittest.TestCase):
    def test_report_ranks_by_velocity_and_escapes_pipes(self):
        repos = [repo("slow/old", "a | b", stars=1000, age_days=100),
                 repo("fast/new", "quick", stars=500, age_days=1)]
        text = render_markdown(repos, find_themes(repos, k=1), NOW, days=30)

        self.assertLess(text.index("fast/new"), text.index("slow/old"))
        self.assertIn("a \\| b", text)

    def test_language_share(self):
        repos = [repo("a/a", ""), repo("b/b", ""), repo("c/c", "", language="Go")]
        self.assertEqual(language_share(repos), [("Python", 2), ("Go", 1)])


if __name__ == "__main__":
    unittest.main()
