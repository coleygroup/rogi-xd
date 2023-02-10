from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto
import os
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Protocol, Type, Union

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


class Fingerprint(AutoName):
    MORGAN = auto()
    TOPOLOGICAL = auto()


class Metric(AutoName):
    DICE = auto()
    TANIMOTO = auto()
    EUCLIDEAN = auto()
    COSINE = auto()
    CITYBLOCK = auto()
    MAHALANOBIS = auto()
    PRECOMPUTED = auto()


class FingerprintConfig(NamedTuple):
    fp: Fingerprint = Fingerprint.MORGAN
    radius: int = 2
    length: int = 2048


class ClassRegistry(Mapping[str, Type]):
    def __init__(self):
        self.__registry = {}

    def register(self, alias: Union[str, Iterable[str], None] = None):
        def decorator(cls):
            if alias is None:
                keys = [cls.__name__.lower()]
            elif isinstance(alias, str):
                keys = [alias]
            else:
                keys = alias

            # class wrapped(cls):
            #     alias = keys[0]

            cls.alias = keys[0]
            for k in keys:
                self.__registry[k] = cls

            return cls

        return decorator

    __call__ = register

    def __getitem__(self, key: str) -> Type:
        return self.__registry[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__registry)

    def __len__(self) -> int:
        return len(self.__registry)


class Configurable(Protocol):
    def to_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        pass


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
        fmt = lambda xs: ", ".join(f"{x:{format_spec}}" for x in xs)
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
