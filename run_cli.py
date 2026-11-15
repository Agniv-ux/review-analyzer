"""Run the analyzer from the terminal, without the web page.

    python run_cli.py --sample
    python run_cli.py "https://play.google.com/store/apps/details?id=com.spotify.music"
    python run_cli.py com.spotify.music 500

Saves the report and the labelled reviews in the "output" folder.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from analyzer.fetch import (SAMPLE_APP_NAME, fetch_app_name, fetch_reviews,
                            get_app_id, load_sample)
from analyzer.llm import LLM
from analyzer.pipeline import analyze
from analyzer.report import to_markdown


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    load_dotenv()  # reads LLM_API_KEY and friends from the .env file
    llm = LLM()

    if sys.argv[1] == "--sample":
        name, app_name, raw = "sample", SAMPLE_APP_NAME, load_sample()
    else:
        name = get_app_id(sys.argv[1])
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        print(f"Fetching up to {count} reviews for {name} ...")
        raw = fetch_reviews(name, count)
        app_name = fetch_app_name(name)

    report, labelled, stats = analyze(
        raw, llm, app_name, progress=lambda fraction, message: print(f"  {message}")
    )

    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / f"{name}_report.md").write_text(to_markdown(report), encoding="utf-8")
    labelled.to_csv(out / f"{name}_labelled.csv", index=False)

    print(f"\nAnalysed {stats['kept']} of {stats['fetched']} reviews "
          f"using {llm.calls} LLM calls.")
    print(f"Report saved to {out / f'{name}_report.md'}")


if __name__ == "__main__":
    main()
