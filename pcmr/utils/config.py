from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    def to_config(self) -> Config:
        pass

    @classmethod
    def from_config(cls, config: Config) -> Configurable:
        pass


@dataclass
class Config:
    def save(self, path):
        d = {
            k: v.to_config() if isinstance(v, Configurable) else v
            for k, v in asdict(self).items()
        }

        path = Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(path) -> Config:
        pass 
