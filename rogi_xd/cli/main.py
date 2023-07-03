from argparse import ArgumentParser
import logging
from typing import Type

from rogi_xd.cli.utils.command import Subcommand
from rogi_xd.cli.finetune import FinetuneSubcommand
from rogi_xd.cli.list import ListSubcommand
from rogi_xd.cli.rogi import RogiSubcommand
from rogi_xd.cli.knn import KnnSubcommand
from rogi_xd.cli.train import TrainSubcommand
from rogi_xd.cli.utils import NOW, pop_attr

logger = logging.getLogger(__name__)

DEFAULT_LOGFILE = f"logs/{NOW}.log"
LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    parent = ArgumentParser(add_help=False)
    parent.add_argument(
        "--logfile",
        "--log",
        nargs="?",
        const=DEFAULT_LOGFILE,
        help="the path to which the log file should be written. Not specifying will this log to stdout. Adding just the flag ('--log/--logfile') will automatically log to a file at 'logs/YYYY-MM-DDTHH:MM:SS.log'",
    )
    parent.add_argument("-v", "--verbose", action="count", default=0, help="the verbosity level")

    parents = [parent]
    SUBCOMMANDS: list[Type[Subcommand]] = [
        ListSubcommand,
        RogiSubcommand,
        KnnSubcommand,
        TrainSubcommand,
        FinetuneSubcommand,
    ]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    args = parser.parse_args()
    logfile, verbose, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "verbose", "mode", "func"]
    )

    logging.basicConfig(
        filename=logfile,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)],
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)
