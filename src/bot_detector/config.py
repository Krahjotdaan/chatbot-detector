from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]
    project_root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.project_root / p


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, project_root=PROJECT_ROOT)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
