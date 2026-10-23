from __future__ import annotations

import httpx


class OllamaUnavailable(RuntimeError):
    pass


class OllamaDDLModel:
    name = "ollama"

    def __init__(self, base_url: str, model_name: str, timeout: float = 30.0, policy_variant: str = "common") -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.policy_variant = policy_variant
        self.version = model_name

    def generate(self, prompt: str, max_new_tokens: int = 192) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": max_new_tokens},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise OllamaUnavailable(
                f"Ollama baseline `{self.model_name}` is unavailable. Start Ollama and pull the model explicitly."
            ) from exc
        payload = response.json()
        return str(payload.get("response", "")).strip()
