from argparse import ArgumentParser
from datetime import datetime
import logging

from pcmr.cli.list import ListSubcommand
from pcmr.cli.rogi import RogiSubcommand
from pcmr.cli.train import TrainSubcommand
from pcmr.cli.utils import pop_attr

logger = logging.getLogger(__name__)

DEFAULT_LOGFILE = f"logs/{datetime.now().isoformat(timespec='seconds')}.log"
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
        help="the path to which the log file should be written. Not specifying will this log to stdout. Adding just the flag ('--log/--logfile') will automatically log a file at 'logs/YYYY-MM-DDTHH:MM:SS.log'",
    )
    parent.add_argument("-v", "--verbose", action="count", default=0, help="the verbosity level")

    parents = [parent]
    ListSubcommand.add(subparsers, parents)
    RogiSubcommand.add(subparsers, parents)
    TrainSubcommand.add(subparsers, parents)

    args = parser.parse_args()
    logfile, verbose, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "verbose", "mode", "func"]
    )

    logging.basicConfig(
        filename=logfile,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)],
        datefmt="%Y-%m-%dT%M:%H:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)
