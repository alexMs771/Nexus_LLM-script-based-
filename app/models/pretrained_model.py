"""Compatibility module: the supported pretrained baseline is Ollama.

The scratch experiment never imports this adapter and never downloads
pretrained weights.
"""

from app.models.ollama_model import OllamaDDLModel as PretrainedOllamaModel

__all__ = ["PretrainedOllamaModel"]
