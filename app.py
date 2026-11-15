"""The web page. Start it with:  streamlit run app.py"""

import os

import streamlit as st
from dotenv import load_dotenv

from analyzer.fetch import (SAMPLE_APP_NAME, fetch_app_name, fetch_reviews,
                            get_app_id, load_sample)
from analyzer.llm import LLM, LLMError
from analyzer.pipeline import analyze
from analyzer.report import SECTIONS, to_markdown

load_dotenv()  # reads LLM_API_KEY and friends from the .env file

st.set_page_config(page_title="Review Analyzer", page_icon="📋", layout="centered")


def get_setting(name: str) -> str | None:
    """Read a setting from .env, or from Streamlit secrets when deployed online."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets.get(name)
    except Exception:
        return None  # no secrets file, which is normal on your own computer


def stars(rating: int) -> str:
    return "1 star" if rating == 1 else f"{rating} stars"


def show_theme(number: int, theme: dict, expanded: bool) -> None:
    """One theme as a click-to-open row: the title shows the size, the body the detail."""
    title = f"{number}. {theme['name']} · {theme['count']} reviews ({theme['share']:.0%})"
    with st.expander(title, expanded=expanded):
        st.caption(f"Average rating of these reviews: {theme['avg_rating']} / 5")
        if theme["suggestion"]:
            st.markdown(f"**Suggested next step:** {theme['suggestion']}")
        for quote in theme["quotes"]:
            st.markdown(f"> {quote['text']}  \n> — {stars(quote['rating'])}")


def show_report(report: dict, labelled, stats: dict) -> None:
    st.header(report["app_name"])
    problem_reviews = sum(t["count"] for t in report["problems"])
    left, middle, right = st.columns(3)
    left.metric("Reviews analysed", report["total_reviews"])
    middle.metric("Average rating", f"{report['avg_rating']} / 5")
    right.metric(
        "Reporting a problem",
        f"{problem_reviews / report['total_reviews']:.0%}",
    )
    removed = stats["fetched"] - stats["kept"]
    period = (
        f"Reviews from {report['date_from']} to {report['date_to']}. "
        if report["date_from"] else ""
    )
    st.caption(
        f"{period}{removed} of {stats['fetched']} fetched reviews were left out "
        "as duplicates, too short, or not in English."
    )

    # One tab per section, so the page stays short. The number is the theme count.
    tabs = st.tabs([f"{title} ({len(report[key])})" for key, title in SECTIONS])
    for tab, (key, title) in zip(tabs, SECTIONS):
        with tab:
            if not report[key]:
                st.write("Nothing stood out in this group.")
            for number, theme in enumerate(report[key], start=1):
                show_theme(number, theme, expanded=(number == 1))

    if report["ungrouped"]:
        st.caption(
            f"{report['ungrouped']} reviews were too vague or one-off to fit a theme."
        )

    st.divider()
    left, right = st.columns(2)
    left.download_button(
        "Download report (Markdown)", to_markdown(report),
        file_name="review_report.md", use_container_width=True,
    )
    right.download_button(
        "Download labelled reviews (CSV)", labelled.to_csv(index=False),
        file_name="labelled_reviews.csv", use_container_width=True,
    )


st.title("Review Analyzer")
st.write(
    "Paste a Play Store link. The AI reads the app's recent reviews and tells "
    "you what to fix, what to build next, and what users love."
)

link = st.text_input(
    "Play Store link or app ID",
    placeholder="https://play.google.com/store/apps/details?id=com.example.app",
)
count = st.slider("How many recent reviews to read", 100, 1000, 300, step=100)
use_sample = st.checkbox(
    "Use sample data instead (made-up reviews for a fictional fitness app)"
)

if st.button("Analyse reviews", type="primary"):
    bar = st.progress(0.0, text="Starting...")
    try:
        llm = LLM(
            api_key=get_setting("LLM_API_KEY"),
            base_url=get_setting("LLM_BASE_URL"),
            model=get_setting("LLM_MODEL"),
        )
        if use_sample:
            app_name, raw = SAMPLE_APP_NAME, load_sample()
        else:
            app_id = get_app_id(link)
            bar.progress(0.02, text="Fetching reviews from the Play Store...")
            raw = fetch_reviews(app_id, count)
            app_name = fetch_app_name(app_id)

        st.session_state["result"] = analyze(
            raw, llm, app_name,
            progress=lambda fraction, message: bar.progress(fraction, text=message),
        )
        bar.empty()
    except (ValueError, RuntimeError, LLMError) as error:
        bar.empty()
        st.session_state.pop("result", None)
        st.error(str(error))

# Kept in session_state so the report stays on screen after a download click.
if "result" in st.session_state:
    show_report(*st.session_state["result"])
