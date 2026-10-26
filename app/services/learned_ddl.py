from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.models.scratch_model import ScratchSQLModel


@lru_cache(maxsize=1)
def get_own_ddl_model(artifact_dir: str | None = None) -> ScratchSQLModel:
    """Load the learned DDL checkpoint only; never call the phase-1 compiler."""

    return ScratchSQLModel.from_dir(Path(artifact_dir) if artifact_dir else settings.own_ddl_model_dir)


def clear_own_ddl_model_cache() -> None:
    get_own_ddl_model.cache_clear()
