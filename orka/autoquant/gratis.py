"""Gratis-only OpenAI-compatible client for AutoQuant reasoning."""
from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


class GratisError(RuntimeError):
    pass


class GratisClient:
    def __init__(self, base_url: str, model: str, max_usd: float, *, post=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self._post = post or self._request

    def _request(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise GratisError(f"Gratis request failed: {exc}") from exc

    def next_action(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._post("/v1/chat/completions", {
            "model": self.model, "messages": messages, "tools": tools, "tool_choice": "required",
        })
        cost = float(response.get("usage", {}).get("cost", 0.0))
        if self.spent_usd + cost > self.max_usd:
            raise GratisError("Gratis budget exhausted")
        self.spent_usd += cost
        choices = response.get("choices", [])
        calls = choices[0].get("message", {}).get("tool_calls", []) if choices else []
        if len(calls) != 1:
            raise GratisError("Gratis response must contain exactly one tool action")
        return calls[0]
