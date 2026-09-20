"""Minimal client for an OpenAI-compatible llama.cpp server with JSON-schema output."""

import json
import os
import time
from typing import Any

import httpx


class LLMClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def model_id(self) -> str:
        resp = self.client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.7,
        seed: int = 42,
        max_tokens: int = 512,
        enable_thinking: bool | None = False,
        retries: int = 3,
    ) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
        """Return (parsed JSON or None, raw text, usage)."""
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        }
        if enable_thinking is not None:
            # Qwen3 hybrid thinking switch (ignored by templates without it)
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.client.post(f"{self.base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                try:
                    return json.loads(raw), raw, usage
                except json.JSONDecodeError:
                    return None, raw, usage
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                # a truncated or malformed body must not kill a 45-minute generation run
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_exc}")
