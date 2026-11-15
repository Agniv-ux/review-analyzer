# Review Analyzer

Paste a Play Store link. The AI reads the app's recent reviews and produces a
report: top problems, most requested features, and what users love, each with
review counts, real quotes, and a suggested next step.

## Setup (once)

These commands are for the VS Code terminal on Windows. On Mac or Linux, use
`python3` instead of `py`.

1. **Install Python** from python.org if `py --version` does not print a
   version. Tick "Add python.exe to PATH" in the installer, then restart VS Code.

2. **Install the libraries:**

   ```
   py -m pip install -r requirements.txt
   ```

3. **Get an LLM API key.** The project is set up for Google's Gemini API, which
   has a free tier. Create a key at https://aistudio.google.com/app/apikey

4. **Save the key.** Make a copy of `.env.example`, name the copy `.env`, and
   replace `paste-your-key-here` with your key. Keep `.env` private: anyone
   with the key can use your account.

## Run it

Start the web page:

```
py -m streamlit run app.py
```

It opens in your browser. Tick **Use sample data** for the first run: it uses
made-up reviews that ship with the project, so you can check your API key works
before involving the Play Store. Then untick it and paste a real app link.

To run without the web page:

```
py run_cli.py --sample
py run_cli.py "https://play.google.com/store/apps/details?id=com.spotify.music"
```

This saves the report and the labelled reviews in an `output` folder.

To check that your API key and model work (makes one small request and never
prints the key):

```
py test_key.py
```

To run the tests (no API key or internet needed):

```
py -m pytest
```

## How it works

Each step is one file in the `analyzer` folder.

| Step | File | What it does |
|---|---|---|
| 1. Fetch | `fetch.py` | Downloads the newest reviews for an app from the Play Store |
| 2. Clean | `clean.py` | Drops empty, duplicate, very short and non-English reviews |
| 3. Label | `label.py` | Sends reviews to the LLM 20 at a time; each gets a type (bug, complaint, feature request, praise), a short topic and a sentiment |
| 4. Themes | `themes.py` | Asks the LLM to merge similar topics into at most 12 themes |
| 5. Report | `report.py` | Counts reviews per theme, picks quotes, asks the LLM for one next step per theme |

Other files:

- `analyzer/llm.py` talks to the LLM API, retries when it is busy, and reads
  the JSON reply.
- `analyzer/pipeline.py` runs steps 2 to 5 in order.
- `app.py` is the web page. `run_cli.py` is the terminal version.
- `tests/` checks the pipeline using a keyword-based stand-in for the LLM.
- `sample_data/sample_reviews.csv` holds made-up reviews for a fictional app.

## Using a different LLM provider

Any provider with an OpenAI-compatible API works. Change `LLM_BASE_URL` and
`LLM_MODEL` in `.env`. Model names change often, so if you get a "model not
found" error, check your provider's current model list.

## Limits to know about

- **Free-tier speed.** 300 reviews need about 17 LLM calls. Free tiers limit
  calls per minute, so the app waits and retries when it hits the limit. A run
  can take a few minutes.
- **Scraping.** Reviews are fetched with an unofficial library, which can break
  when Google changes its pages and may conflict with the store's terms of
  service. Fine for a prototype; move to official access before charging users.
- **Accuracy.** The LLM makes mistakes. Before sending a report to a developer,
  open the labelled CSV and read 30 to 50 rows to check the labels make sense.
- **English only, Play Store only** in this version.
