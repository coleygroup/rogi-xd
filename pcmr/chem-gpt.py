import argparse
from pathlib import Path

import torch
from transformers import pipeline
import selfies as sf

from pcmr.data.molnet import get_task_data
from pcmr.rogi import rogi
from pcmr.utils import Dataset, Metric

CHEMGPT =  "ncfrey/ChemGPT-1.2B"


def main(args):
    featurizer = pipeline(
        "feature-extraction", model=CHEMGPT, device=0, framework="pt", return_tensors=True
    )
    featurizer.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    smis, y = get_task_data(args)

    sfs = [sf.encoder(smi) for smi in smis]
    output = featurizer(sfs)
    X = torch.stack([H[0, -1, :] for H in output]).numpy()

    score, _ = rogi(y, True, X, metric=Metric.EUCLIDEAN)
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=Dataset.keys())
    parser.add_argument("-t", "--task")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-b", "--batch-size", type=int, default=8)

    args = parser.parse_args()
    args.dataset = Dataset.get(args.dataset)

    main(args)