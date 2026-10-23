"""Whisper-JAX Tibetan-Chinese Speech-to-Text & Translation Toolkit."""

from src.compat import preload_liblzma
from src.pipeline import run_pipeline

__all__ = ["preload_liblzma", "run_pipeline"]
