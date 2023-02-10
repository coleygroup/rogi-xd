from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Union

import requests
import torch
from tqdm import tqdm

CACHE_DIR = os.environ.get("PCMR_CACHE", Path.home() / ".cache" / "pcmr")


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unsupported alias! got: {name}. expected one of: {cls.keys()}")

    @classmethod
    def keys(cls) -> list[str]:
        return [e.value for e in cls]


# @dataclass
# class Factory:
#     registry: ClassRegistry

#     def to_config(self, obj: Configurable):
#         return {"alias": obj.alias, "config": obj.to_config()}

#     def from_config(self, config):
#         alias = config["v_reg"]["alias"]
#         cls_config = config["v_reg"]["config"]

#         return self.registry[alias].from_config(cls_config)


class flist(list):
    def __format__(self, format_spec):
        fmt = lambda xs: ", ".join(f"{x:{format_spec}}" for x in xs)    # noqa: E731
        if len(self) >= 6:
            s = f"[{fmt(self[:3])}, ..., {fmt(self[-3:])}]"
        else:
            s = f"[{fmt(self)}]"

        return s

    def __str__(self) -> str:
        return f"{self}"


def select_device(device: Union[int, str, torch.device, None]):
    return device or (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")


def download_file(url: str, path: Path, desc: str = "Downloading", chunk_size: int = 1024):
    """download the file at the specified URL to the indicated path"""

    with (
        requests.get(url, stream=True) as response,
        open(path, "wb") as fid,
        tqdm(desc=desc, unit="B", unit_scale=True, unit_divisor=chunk_size, leave=False) as bar,
    ):
        bar.reset(int(response.headers["content-length"]))
        for chunk in response.iter_content(chunk_size):
            fid.write(chunk)
            bar.update(chunk_size)
