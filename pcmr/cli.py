from argparse import ArgumentParser
import logging
from pathlib import Path

from pcmr.data import data
from pcmr.featurizers import FeaturizerRegistry
from pcmr.rogi import rogi
from pcmr.utils import Metric


def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s:%(name)s - %(message)s', level=logging.INFO
    )
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--featurizer", type=lambda s: s.lower(), choices=FeaturizerRegistry.keys()
    )
    parser.add_argument("-d", "--dataset", type=lambda s: s.upper(), choices=data.datasets)
    parser.add_argument("-t", "--task")
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    args = parser.parse_args()

    score = calc_rogi(args)

    print(score)

def calc_rogi(args):
    f = FeaturizerRegistry[args.featurizer]()
    df = data.get_data(args.dataset, args.task, args.n)

    X = f(df.smiles.tolist())
    score, _ = rogi(df.y.tolist(), True, X, metric=Metric.EUCLIDEAN, min_dt=0.01)
    return score