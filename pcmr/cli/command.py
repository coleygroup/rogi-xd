from abc import ABC, abstractmethod
from argparse import ArgumentParser, _SubParsersAction, Namespace
from typing import Optional


class Subcommand(ABC):
    COMMAND: str
    HELP: Optional[str] = None

    @classmethod
    def add(cls, subparsers: _SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help=cls.HELP)
        cls.add_args(parser).set_defaults(func=cls.func)

        return parser

    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        pass

    @staticmethod
    @abstractmethod
    def func(args: Namespace):
        pass
