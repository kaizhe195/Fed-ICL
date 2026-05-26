"""Language model client interfaces.大模型选择"""

from __future__ import annotations

import hashlib
import json
import random
import re
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

ANSWER_LETTERS = ["A", "B", "C", "D"]


class LMClient(ABC):
    """Base interface for local language model clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a completion for a prompt."""


def parse_mmlu_answer(text: str | None) -> Optional[str]:
    """Parse a model response into A, B, C, or D."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    single_letter = re.fullmatch(r"\s*([ABCDabcd])\s*[\.\)]?\s*", stripped)
    if single_letter:
        return single_letter.group(1).upper()

    answer_patterns = [
        r"\banswer\s*[:\-]\s*[\(\[]?\s*([ABCDabcd])\s*[\)\]\.]?",
        r"\banswer\s+is\s+[\(\[]?\s*([ABCDabcd])\s*[\)\]\.]?",
        r"\bcorrect\s+answer\s+is\s+[\(\[]?\s*([ABCDabcd])\s*[\)\]\.]?",
        r"\bcorrect\s+option\s+is\s+[\(\[]?\s*([ABCDabcd])\s*[\)\]\.]?",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    option_match = re.search(
        r"\b(?:option|choice)\s+[\(\[]?\s*([ABCDabcd])\s*[\)\]\.]?",
        stripped,
        flags=re.IGNORECASE,
    )
    if option_match:
        return option_match.group(1).upper()
    return None


@dataclass
class MockLMClient(LMClient):
    """Deterministic mock model for pipeline verification only."""

    seed: int = 42

    def generate(self, prompt: str) -> str:
        answers = re.findall(r"Answer:\s*([ABCD])\b", prompt)
        if answers:
            counts = {letter: answers.count(letter) for letter in ANSWER_LETTERS}
            max_count = max(counts.values())
            winners = [letter for letter in ANSWER_LETTERS if counts[letter] == max_count]
            if len(winners) == 1:
                return winners[0]

        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        return rng.choice(ANSWER_LETTERS)


@dataclass
class OpenAICompatibleLMClient(LMClient):
    """Skeleton for OpenAI-compatible chat completion APIs."""

    model_name: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI-compatible backend requested, but the 'openai' package is not installed. "
                "Install it and provide llm.api_key or an OPENAI_API_KEY-compatible environment before use."
            ) from exc
        if not self.model_name:
            raise RuntimeError("OpenAI-compatible backend requested, but llm.model_name is empty.")
        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        client = OpenAI(**client_kwargs)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenAI-compatible backend request failed. Check llm.base_url, credentials, "
                "network access, and model availability."
            ) from exc
        content = response.choices[0].message.content
        return content or ""


@dataclass
class OllamaCompatibleLMClient(LMClient):
    """Local Ollama-compatible generation client."""

    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 8
    timeout_seconds: int = 120

    def generate(self, prompt: str) -> str:
        if not self.model_name:
            raise RuntimeError("Ollama backend requested, but llm.model_name is empty.")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = _extract_ollama_error(body)
            if exc.code == 404 or "not found" in message.lower() or "pull" in message.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' is not available. "
                    f"Pull it with: ollama pull {self.model_name}"
                ) from exc
            raise RuntimeError(
                f"Ollama backend returned HTTP {exc.code}. {message or 'Check the Ollama server logs.'}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama backend requested, but the Ollama server is unavailable. "
                f"Start Ollama and verify the endpoint: {self.base_url}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout_seconds} seconds. "
                "Use a smaller model, increase llm.timeout_seconds, or check the local server."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout_seconds} seconds. "
                "Use a smaller model, increase llm.timeout_seconds, or check the local server."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response from /api/generate.") from exc
        except Exception as exc:
            raise RuntimeError(
                "Ollama backend request failed. Check llm.base_url and local model availability."
            ) from exc
        if "error" in data:
            message = str(data["error"])
            if "not found" in message.lower() or "pull" in message.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' is not available. "
                    f"Pull it with: ollama pull {self.model_name}"
                )
            raise RuntimeError(f"Ollama backend error: {message}")
        return str(data.get("response", ""))


def _extract_ollama_error(body: str) -> str:
    """Extract an Ollama error message from an HTTP response body."""
    if not body:
        return ""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()
    if isinstance(data, dict):
        return str(data.get("error", "")).strip()
    return body.strip()


OpenAICompatibleClient = OpenAICompatibleLMClient
OllamaCompatibleClient = OllamaCompatibleLMClient


def create_lm_client(config: dict, seed: int = 42) -> LMClient:
    """Construct an LM client from config."""
    backend = config.get("backend", "mock")
    if backend == "mock":
        return MockLMClient(seed=seed)
    if backend in {"openai", "openai_compatible"}:
        return OpenAICompatibleLMClient(
            model_name=config.get("model_name", ""),
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 8)),
        )
    if backend in {"ollama", "ollama_compatible"}:
        return OllamaCompatibleLMClient(
            model_name=config.get("model_name", ""),
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_tokens", 8)),
            timeout_seconds=int(config.get("timeout_seconds", 120)),
        )
    raise ValueError(
        f"Unsupported LLM backend: {backend}. Supported backends are: mock, openai, openai_compatible, "
        "ollama, ollama_compatible."
    )
