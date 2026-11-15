"""A small wrapper around the LLM API.

It uses the `openai` library, but works with any provider that offers an
OpenAI-compatible API (Gemini, Groq, OpenAI, OpenRouter, ...). Which one is
used depends on three settings in the .env file:

    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
"""

import json
import os
import time

from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI, RateLimitError

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

RETRY_WAITS = [5, 20, 40]  # seconds to wait before each retry


class LLMError(RuntimeError):
    """Raised when the LLM cannot give us a usable answer."""


def parse_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply.

    Models sometimes wrap JSON in ```json fences or add a sentence around it,
    so we take everything from the first "{" to the last "}".
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("the reply was JSON but not an object")
    return data


class LLM:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None):
        api_key = api_key or os.getenv("LLM_API_KEY")
        if not api_key or api_key == "paste-your-key-here":
            raise LLMError(
                "No API key found. Copy .env.example to .env and put your key "
                "in LLM_API_KEY (see the README)."
            )
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL,
            max_retries=0,  # we do our own retries below
            timeout=120,
        )
        self.json_mode = True  # ask the API to guarantee JSON, if it can
        self.calls = 0

    def ask_json(self, system: str, user: str) -> dict:
        """Send one prompt and return the reply as a Python dict."""
        last_error = None
        for attempt in range(len(RETRY_WAITS) + 1):
            if attempt:
                time.sleep(RETRY_WAITS[attempt - 1])
            try:
                return parse_json(self._call(system, user))
            except (RateLimitError, APIConnectionError) as error:
                last_error = error  # busy or offline: wait and retry
            except APIStatusError as error:
                if error.status_code < 500:
                    raise LLMError(f"The LLM API rejected the request: {error}") from error
                last_error = error  # server problem: wait and retry
            except ValueError as error:
                last_error = error  # reply was not valid JSON: ask again
        raise LLMError(f"The LLM did not give a usable answer: {last_error}")

    def _call(self, system: str, user: str) -> str:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        self.calls += 1
        if self.json_mode:
            try:
                reply = self.client.chat.completions.create(
                    **request, response_format={"type": "json_object"}
                )
                return reply.choices[0].message.content or ""
            except BadRequestError:
                self.json_mode = False  # this provider has no JSON mode
        reply = self.client.chat.completions.create(**request)
        return reply.choices[0].message.content or ""
