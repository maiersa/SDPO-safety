"""Shared CLI helpers for OPSD alignment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to an OPSD alignment YAML config.")


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load OPSD alignment configs.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def output_path(config: dict[str, Any], *parts: str) -> Path:
    output_dir = Path(config["paths"]["output_dir"])
    return output_dir.joinpath(*parts)

