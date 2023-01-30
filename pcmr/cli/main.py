from argparse import ArgumentParser
import logging
from pcmr.cli.list import ListSubcommand
from pcmr.cli.rogi import RogiSubcommand

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s", level=logging.INFO
    )

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    ListSubcommand.add(subparsers)
    RogiSubcommand.add(subparsers)

    args = parser.parse_args()
    logger.info(f"Running with args: {args}")
    args.func(args)
