from __future__ import annotations

from functools import lru_cache
import httpx

from fastapi import HTTPException

from app.config import settings
from app.models.ollama_model import OllamaDDLModel
from app.models.scratch_model import ScratchSQLModel
from app.services.learned_sql import get_own_sql_model


@lru_cache(maxsize=1)
def loaded_models() -> dict[str, object]:
    models: dict[str, object] = {}
    try:
        models["scratch"] = ScratchSQLModel.from_dir(settings.scratch_dir)
    except FileNotFoundError:
        pass
    return models


def clear_model_cache() -> None:
    loaded_models.cache_clear()


def ollama_available() -> bool:
    try:
        response = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2.0)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def health_model_label() -> str:
    names = sorted(name for name in loaded_models() if name != "ollama")
    return ",".join(names) if names else "not_loaded"


def select_model(model_name: str):
    models = loaded_models()
    if model_name == "ollama":
        # Remote baseline availability is intentionally checked only when used.
        return OllamaDDLModel(settings.ollama_url, settings.ollama_model, settings.ollama_timeout_seconds)
    if model_name in models:
        return models[model_name]
    raise HTTPException(
        status_code=503,
        detail="Scratch model is not loaded. Run `python -m training.train_scratch` first.",
    )


def model_inventory() -> list[dict]:
    models = loaded_models()
    return [
        {
            "id": "own-sql",
            "provider": "local_pytorch",
            "available": _own_sql_available(),
            "version": "own-sql-v1" if _own_sql_available() else "",
            "details": "Random-initialized decoder-only Transformer trained from the prepared NL-to-SQL corpus; no compiler fallback.",
        },
        {
            "id": "scratch",
            "provider": "local_pytorch_hybrid",
            "available": "scratch" in models,
            "version": "scratch-checkpoint" if "scratch" in models else "",
            "details": "Random-initialized custom decoder-only Transformer with a validated local schema planner for table requests.",
        },
        {
            "id": "ollama",
            "provider": "ollama",
            "available": ollama_available(),
            "version": settings.ollama_model,
            "details": "Local Ollama model selected by default; the configured model must be installed and Ollama must be running.",
        },
    ]


def _own_sql_available() -> bool:
    try:
        get_own_sql_model()
        return True
    except FileNotFoundError:
        return False
