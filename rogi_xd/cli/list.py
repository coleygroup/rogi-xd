from argparse import ArgumentParser, FileType, Namespace

from rogi_xd.data import data
from rogi_xd.cli.utils.command import Subcommand


class ListSubcommand(Subcommand):
    COMMAND = "list"
    HELP = "list the available datasets"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("output", nargs="?", type=FileType("w"))

        return parser

    @staticmethod
    def func(args: Namespace):
        for d in data.datasets:
            tasks = data.tasks(d)
            if len(tasks) > 1:
                [print(f"{d}/{t}", file=args.output) for t in tasks]
            else:
                print(d, file=args.output)
