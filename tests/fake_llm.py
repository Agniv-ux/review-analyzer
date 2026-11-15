"""A stand-in for the real LLM, used only by the tests.

It labels reviews with simple keyword rules, so the tests can run the whole
pipeline without an API key or internet. It is far cruder than a real model.
"""

import re

RULES = [  # (keywords, type, topic); the first matching rule wins
    (("calorie",), "bug", "inaccurate calorie counts"),
    (("sync", "watch", "band"), "bug", "crash on smartwatch sync"),
    (("ads", " ad "), "complaint", "too many ads"),
    (("dark",), "feature_request", "dark mode"),
    (("log in", "login", "code"), "bug", "login code not arriving"),
    (("price", "expensive", "subscription"), "complaint", "premium too expensive"),
    (("export", "download all"), "feature_request", "export workout data"),
    (("plan",), "praise", "great workout plans"),
    (("simple", "clean", "easy to use"), "praise", "simple clean design"),
    (("chart", "graph"), "praise", "motivating progress charts"),
]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def ask_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if "label app store reviews" in system:
            return self._label(user)
        if "organise feedback topics" in system:
            return self._merge(user)
        return self._suggest(user)

    def _label(self, user: str) -> dict:
        out = []
        for number, rating, text in re.findall(r"^(\d+)\. \[(\d) stars\] (.*)$", user, re.M):
            lowered = f" {text.lower()} "
            label = next(
                ((t, topic) for words, t, topic in RULES if any(w in lowered for w in words)),
                ("other", "unclear"),
            )
            out.append(
                {
                    "id": int(number),
                    "type": label[0],
                    "topic": label[1],
                    "sentiment": "negative" if int(rating) <= 2 else "positive",
                }
            )
        return {"reviews": out}

    def _merge(self, user: str) -> dict:
        themes = []
        for number, topic in re.findall(r"^(\d+)\. \[\w+\] (.*) \(\d+ reviews\)$", user, re.M):
            themes.append({"name": topic.capitalize(), "topic_ids": [int(number)]})
        return {"themes": themes}

    def _suggest(self, user: str) -> dict:
        found = re.findall(r"^(\d+)\. (.*) \((?:problem|request), \d+ reviews\)$", user, re.M)
        return {
            "suggestions": [
                {"id": int(n), "suggestion": f"Look into '{name.lower()}' first."}
                for n, name in found
            ]
        }
