"""Runs all the steps in order: clean -> label -> merge themes -> report."""

import pandas as pd

from .clean import clean_reviews
from .label import label_reviews
from .report import build_report
from .themes import merge_topics


def analyze(raw_reviews: pd.DataFrame, llm, app_name: str, progress=None):
    """Turn raw reviews into a report.

    `progress` is an optional function called as progress(fraction, message),
    where fraction goes from 0.0 to 1.0.

    Returns (report dict, labelled review table, cleaning stats).
    """
    def tell(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    tell(0.05, "Cleaning reviews...")
    reviews, stats = clean_reviews(raw_reviews)
    if reviews.empty:
        raise ValueError("No usable reviews were left after cleaning.")

    def labelling_progress(done: int, total: int) -> None:
        tell(0.1 + 0.65 * done / total, f"Reading reviews... {done} of {total}")

    labelling_progress(0, len(reviews))  # show this stage before the first batch returns
    labelled = label_reviews(reviews, llm, progress=labelling_progress)

    tell(0.8, "Grouping topics into themes...")
    themed = merge_topics(labelled, llm)

    tell(0.9, "Writing the report...")
    report = build_report(themed, llm, app_name)

    tell(1.0, "Done")
    return report, themed, stats
