"""Step 2: remove reviews that would only add noise."""

import pandas as pd

MIN_WORDS = 4  # "nice", "good app" and similar say nothing we can act on


def _mostly_english(text: str) -> bool:
    """Rough check: at least 80% of the letters are plain a-z letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= 0.8


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned table plus counts of what was removed and why."""
    stats = {"fetched": len(df)}
    df = df.copy()

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["likes"] = pd.to_numeric(df.get("likes", 0), errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    before = len(df)
    df = df[(df["text"] != "") & df["rating"].notna()]
    stats["empty"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="review_id")
    df = df[~df["text"].str.lower().duplicated()]
    stats["duplicates"] = before - len(df)

    before = len(df)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]
    stats["too_short"] = before - len(df)

    before = len(df)
    df = df[df["text"].map(_mostly_english)]
    stats["not_english"] = before - len(df)

    df["rating"] = df["rating"].astype(int)
    stats["kept"] = len(df)
    return df.reset_index(drop=True), stats
