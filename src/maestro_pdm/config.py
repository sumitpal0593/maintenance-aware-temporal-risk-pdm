from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML experiment configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    return config


def get_data_dir() -> Path:
    """
    Resolve the Scania Component X data directory from SCANIA_DATA_DIR.

    Create a local .env file from .env.example before running the project.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    value = os.getenv("SCANIA_DATA_DIR")
    if not value:
        raise EnvironmentError(
            "SCANIA_DATA_DIR is not set. Copy .env.example to .env and "
            "set the path to the Scania Component X data directory."
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SCANIA_DATA_DIR does not exist: {path}")

    return path


def get_output_dir(config: Dict[str, Any]) -> Path:
    """Resolve and create the configured output directory."""
    output_root = Path(config.get("output", {}).get("root_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root
