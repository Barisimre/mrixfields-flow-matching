"""Environment configuration loader for MRIxFields2026.

Reads settings from .env file at the repository root.
All path and device configuration is centralized here.

Relative paths in .env are resolved relative to the .env file location.
"""

import os
from pathlib import Path

_loaded = False
_env_dir: Path | None = None  # Directory containing .env file


def load_env():
    """Load .env file from repository root into os.environ.

    Only loads variables that are not already set in the environment,
    so explicit environment variables always take precedence.
    Silently skips if .env does not exist.
    """
    global _loaded, _env_dir
    if _loaded:
        return
    _loaded = True

    # Search upward from this file for the repo root (the dir containing .env),
    # so the loader is robust to where the package sits in the tree.
    here = Path(__file__).resolve()
    env_path = next((p / ".env" for p in (here.parent, *here.parents)
                     if (p / ".env").exists()), None)
    if env_path is None:
        return

    _env_dir = env_path.parent

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


def _resolve_path(path: str) -> str:
    """Resolve relative paths relative to .env file location."""
    if path and _env_dir and not os.path.isabs(path):
        resolved = (_env_dir / path).resolve()
        return str(resolved)
    return path


def get_data_dir() -> str:
    """Get DATA_DIR from environment."""
    load_env()
    val = os.environ.get("DATA_DIR")
    if not val:
        raise RuntimeError(
            "DATA_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_output_dir() -> str:
    """Get OUTPUT_DIR from environment."""
    load_env()
    val = os.environ.get("OUTPUT_DIR")
    if not val:
        raise RuntimeError(
            "OUTPUT_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_inference_dir() -> str:
    """Get INFERENCE_DIR from environment."""
    load_env()
    val = os.environ.get("INFERENCE_DIR")
    if not val:
        raise RuntimeError(
            "INFERENCE_DIR is not set. Configure it in .env (repo root) "
            "or set the environment variable. See .env.example."
        )
    return _resolve_path(val)


def get_device() -> str:
    """Get DEVICE from environment."""
    load_env()
    return os.environ.get("DEVICE", "cuda:0")
