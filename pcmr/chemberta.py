import argparse
from pathlib import Path

import torch
from transformers import pipeline

from pcmr.data.molnet import get_task_data
from pcmr.rogi import rogi
from pcmr.utils import Dataset, Metric

CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"


def main(args):
    featurizer = pipeline(
        "feature-extraction", model=CHEMBERTA, device=0, framework="pt", return_tensors=True
    )

    smis, y = get_task_data(args)

    output = featurizer(smis, batch_size=args.batch_size)
    X = torch.stack([H[0, 0, :] for H in output]).numpy()

    score, _ = rogi(y, True, X, metric=Metric.EUCLIDEAN)
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=Dataset.keys())
    parser.add_argument("-t", "--task")
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=32)

    args = parser.parse_args()
    args.dataset = Dataset.get(args.dataset)

    main(args)
