from argparse import ArgumentParser
import logging

from pcmr.cli.list import ListSubcommand
from pcmr.cli.rogi import RogiSubcommand

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--logfile", "--log")
    parser.add_argument("-v", "--verbose", action="count", default=0)

    subparsers = parser.add_subparsers()
    ListSubcommand.add(subparsers)
    RogiSubcommand.add(subparsers)

    args = parser.parse_args()

    LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = LOG_LEVELS[min(args.verbose, len(LOG_LEVELS) - 1)]

    logging.basicConfig(
        filename=args.logfile,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=log_level
    )
    
    logger.info(f"Running with args: {args}")
    args.func(args)
