"""Step 4: merge the many small topics into a handful of themes.

The same issue comes back worded differently ("sync crash", "crashes when
syncing watch"). We show the LLM the full list of topics and ask it to group
them, then count how many reviews fall into each group.
"""

import pandas as pd

MAX_THEMES = 12
MAX_TOPICS_IN_PROMPT = 400
OTHER = "Other"

SYSTEM_PROMPT = """You organise feedback topics from app reviews into themes.

You get a numbered list of topics. Each has a type in brackets and the number of
reviews that mention it. Group topics that describe the same underlying issue,
request or strength into one theme.

Rules:
- Create at most {max_themes} themes. Fewer is fine.
- Keep each theme to one kind of feedback: do not mix praise with problems, and
  keep feature requests separate from bugs and complaints.
- Give each theme a short, specific name (2 to 6 words) that a developer would
  understand at a glance, such as "Smartwatch sync crashes" or "Too many ads".
- Put each topic in at most one theme. Leave out topics that fit nowhere.

Reply with JSON only, in this shape:
{{"themes": [{{"name": "Smartwatch sync crashes", "topic_ids": [1, 5, 9]}}]}}"""


def merge_topics(df: pd.DataFrame, llm) -> pd.DataFrame:
    """Add a "theme" column to the labelled review table."""
    df = df.copy()
    usable = df[(df["type"] != "other") & (df["topic"] != "unclear")]
    if usable.empty:
        df["theme"] = OTHER
        return df

    # One row per (type, topic) with its review count, biggest first.
    topics = (
        usable.groupby(["type", "topic"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(MAX_TOPICS_IN_PROMPT)
        .reset_index(drop=True)
    )
    listing = "\n".join(
        f"{i + 1}. [{row.type}] {row.topic} ({row.count} reviews)"
        for i, row in enumerate(topics.itertuples())
    )
    reply = llm.ask_json(
        SYSTEM_PROMPT.format(max_themes=MAX_THEMES), "Topics:\n\n" + listing
    )

    theme_of: dict[tuple[str, str], str] = {}
    themes = reply.get("themes", [])
    for theme in themes if isinstance(themes, list) else []:
        name = " ".join(str(theme.get("name", "")).split())
        ids = theme.get("topic_ids", [])
        if not name or not isinstance(ids, list):
            continue
        for topic_id in ids:
            try:
                row = topics.iloc[int(topic_id) - 1]
            except (IndexError, TypeError, ValueError):
                continue
            theme_of.setdefault((row["type"], row["topic"]), name)  # first theme wins

    df["theme"] = [
        theme_of.get((review_type, topic), OTHER)
        for review_type, topic in zip(df["type"], df["topic"])
    ]
    return df
