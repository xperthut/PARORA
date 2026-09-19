"""
Shared local-model configuration for all three PARORA entry points.

Reads config.yaml (next to this file) once per process. Precedence, highest
wins: environment variable > config.yaml > hardcoded fallback below --
matching every other override already in this repo (OLLAMA_HOST,
PARORA_LOG_LEVEL, ...). Edit config.yaml to point an entry point at a
different local Ollama model, or tune its context window / keep-alive,
without touching server.py, app_lite.py or app.py.
"""

import os
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

try:
    with open(_CONFIG_PATH) as f:
        _RAW = yaml.safe_load(f) or {}
except FileNotFoundError:
    _RAW = {}

_DEFAULTS = _RAW.get("defaults", {})


def _resolve_host(host: str) -> str:
    """Docker bridge fallback -- same rule app.py/app_lite.py applied inline before this module existed."""
    if os.path.exists("/.dockerenv") and "OLLAMA_HOST" not in os.environ:
        return "http://host.docker.internal:11434"
    return host


def get_config(entry_point: str) -> dict:
    """
    Resolved Ollama settings for one entry point: "server", "app_lite" or
    "app". Returns model, ollama_host, temperature, num_ctx, keep_alive.
    """
    section = _RAW.get(entry_point, {})
    host = os.getenv("OLLAMA_HOST", _RAW.get("ollama_host", "http://localhost:11434"))
    return {
        "model": os.environ.get(
            f"PARORA_MODEL_{entry_point.upper()}",
            section.get("model", "qwen2.5:7b")),
        "ollama_host": _resolve_host(host),
        "temperature": float(os.environ.get(
            "PARORA_TEMPERATURE", _DEFAULTS.get("temperature", 0.0))),
        "num_ctx": int(os.environ.get(
            "PARORA_NUM_CTX", _DEFAULTS.get("num_ctx", 16384))),
        "keep_alive": os.environ.get(
            "PARORA_KEEP_ALIVE", _DEFAULTS.get("keep_alive", "30m")),
    }
