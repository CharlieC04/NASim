from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

SECTIONS = (
    "geometry",
    "interaction",
    "transport",
    "addressing",
    "readout",
    "gates",
    "coherence"
)

def load_config(path: str | Path) -> dict[str, Any]:

    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} could not be parsed")

    missing = [section for section in SECTIONS if section not in data]
    if missing:
        raise ValueError(f"Config at {path} was missing sections {missing}")

    return data