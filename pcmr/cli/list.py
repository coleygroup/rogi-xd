from argparse import ArgumentParser, FileType, Namespace

from pcmr.data import data
from pcmr.cli.command import Subcommand

def center(word, fill: str = '=', width: int = 30):
    return f"{word:{fill}^{width}}"


def list_all(args):
    for d in data.datasets:
        ts = data.get_tasks(d)
        if len(ts) > 1:
            [print(f"{d}/{t}", file=args.output) for t in ts]
        else:
            print(d, file=args.output)


class ListSubcommand(Subcommand):
    COMMAND = "list"
    # FUNC = list_all

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("output", nargs="?", type=FileType("w"))

        return parser
    
    @staticmethod
    def func(args: Namespace):
        for d in data.datasets:
            ts = data.get_tasks(d)
            if len(ts) > 1:
                [print(f"{d}/{t}", file=args.output) for t in ts]
            else:
                print(d, file=args.output)