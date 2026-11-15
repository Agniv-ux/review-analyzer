"""Step 1: get the reviews (from the Play Store, or from the sample file)."""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

SAMPLE_FILE = Path(__file__).parent.parent / "sample_data" / "sample_reviews.csv"
SAMPLE_APP_NAME = "FitTrack (sample data)"


def get_app_id(link_or_id: str) -> str:
    """Accept a full Play Store link or a bare app ID and return the app ID.

    https://play.google.com/store/apps/details?id=com.spotify.music -> com.spotify.music
    """
    text = link_or_id.strip()
    if not text:
        raise ValueError("Please enter a Play Store link or an app ID.")
    if "play.google.com" not in text:
        return text  # already an app ID

    query = parse_qs(urlparse(text).query)
    if "id" not in query:
        raise ValueError(f"Could not find an app ID in this link: {text}")
    return query["id"][0]


def fetch_reviews(app_id: str, count: int = 300) -> pd.DataFrame:
    """Download the newest `count` English reviews for one app."""
    # Imported here so the rest of the project works even without this library.
    from google_play_scraper import Sort, reviews

    result, _ = reviews(app_id, lang="en", country="us", sort=Sort.NEWEST, count=count)

    if not result:
        raise RuntimeError(
            f"No reviews came back for '{app_id}'. Check the app ID and your "
            "internet connection. If both are fine, the scraper library may "
            "need an update: pip install --upgrade google-play-scraper"
        )

    df = pd.DataFrame(result)
    df = df[["reviewId", "content", "score", "thumbsUpCount", "at", "appVersion"]]
    return df.rename(
        columns={
            "reviewId": "review_id",
            "content": "text",
            "score": "rating",
            "thumbsUpCount": "likes",
            "at": "date",
            "appVersion": "app_version",
        }
    )


def fetch_app_name(app_id: str) -> str:
    """Look up the app's display name. Falls back to the ID if that fails."""
    try:
        from google_play_scraper import app

        return app(app_id, lang="en", country="us")["title"]
    except Exception:
        return app_id


def load_sample() -> pd.DataFrame:
    """Load the made-up reviews that ship with the project (no internet needed)."""
    return pd.read_csv(SAMPLE_FILE)
