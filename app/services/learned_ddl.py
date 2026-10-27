from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.models.encoder_decoder_model import EncoderDecoderDDLModel
from app.models.scratch_model import ScratchSQLModel


def get_own_ddl_artifact_dir() -> Path:
    """Prefer the trained encoder-decoder artifact, then retain legacy fallback."""
    encoder_decoder = settings.own_ddl_encoder_decoder_dir
    if (encoder_decoder / "config.json").exists():
        return encoder_decoder
    return settings.own_ddl_model_dir


@lru_cache(maxsize=1)
def get_own_ddl_model(artifact_dir: str | None = None) -> ScratchSQLModel | EncoderDecoderDDLModel:
    """Load the learned DDL checkpoint only; never call the phase-1 compiler."""
    path = Path(artifact_dir) if artifact_dir else get_own_ddl_artifact_dir()
    config_path = path / "config.json"
    if config_path.exists():
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("architecture") == "custom_encoder_decoder_transformer":
            return EncoderDecoderDDLModel.from_dir(path)
    return ScratchSQLModel.from_dir(path)


def clear_own_ddl_model_cache() -> None:
    get_own_ddl_model.cache_clear()
