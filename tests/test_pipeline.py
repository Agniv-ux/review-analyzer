"""Run with:  python -m pytest"""

import pandas as pd
import pytest

from analyzer.clean import clean_reviews
from analyzer.fetch import get_app_id, load_sample
from analyzer.llm import parse_json
from analyzer.pipeline import analyze
from analyzer.report import to_markdown
from tests.fake_llm import FakeLLM


def test_get_app_id_from_link_and_bare_id():
    link = "https://play.google.com/store/apps/details?id=com.spotify.music&hl=en"
    assert get_app_id(link) == "com.spotify.music"
    assert get_app_id("  com.spotify.music ") == "com.spotify.music"
    with pytest.raises(ValueError):
        get_app_id("https://play.google.com/store/apps")
    with pytest.raises(ValueError):
        get_app_id("   ")


def test_clean_removes_junk():
    raw = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d", "e", "e"],
            "text": [
                "The app crashes when I open settings",
                "good",
                "",
                "the app crashes when i open settings",  # same text, different case
                "Please add a dark mode option",
                "Please add a dark mode option",
            ],
            "rating": [1, 5, 3, 1, 4, 4],
            "likes": [2, 0, 0, 0, 1, 1],
            "date": ["2026-11-01"] * 6,
            "app_version": ["1.0"] * 6,
        }
    )
    cleaned, stats = clean_reviews(raw)
    assert list(cleaned["review_id"]) == ["a", "e"]
    assert stats == {
        "fetched": 6, "empty": 1, "duplicates": 2,
        "too_short": 1, "not_english": 0, "kept": 2,
    }


def test_parse_json_handles_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": {"b": 2}} hope it helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_full_pipeline_on_sample_data():
    report, labelled, stats = analyze(load_sample(), FakeLLM(), "FitTrack")

    # Every kept review got a label and a theme.
    assert len(labelled) == stats["kept"] == report["total_reviews"]
    assert labelled[["type", "topic", "sentiment", "theme"]].notna().all().all()

    # The biggest problem in the sample data is the smartwatch sync crash.
    top_problem = report["problems"][0]
    assert top_problem["name"] == "Crash on smartwatch sync"
    assert top_problem["count"] == 12
    assert len(top_problem["quotes"]) == 3
    assert top_problem["suggestion"]

    assert report["requests"][0]["name"] == "Dark mode"
    assert report["praise"][0]["name"] == "Great workout plans"

    # Counts add up: every review is in exactly one theme or ungrouped.
    in_themes = sum(t["count"] for k in ("problems", "requests", "praise") for t in report[k])
    assert in_themes + report["ungrouped"] == report["total_reviews"]

    markdown = to_markdown(report)
    assert "## Top problems" in markdown and "Crash on smartwatch sync" in markdown
