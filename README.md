# ROGI-XD

Evaluating the roughness of learned representations from chemical foundation models

## Table of Contents

- [Setup](#setup)
- [Pretrained models](#pretrained-models)
- [Results](#results)
- [Figures](#figures)

## Setup

0. (if necessary) install conda
1. clone this repository and its submodules:
    ```
    git clone --recurse-submodules THIS_REPO
    ```
2. Choose one option below to install all required packages

    a. **Conda** (recommended)
    ```
    conda env create -f environment.yml
    ```
    _Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu` before running the above command
    ```
    sed -i -e 's/cu116/cpu/g' environment.yml
    conda env create -f environment.yml
    ```
    
    b. **Manual**
    
    _Note_: this is only recommended in the event that you'd like specify different package versions
    ```
    conda create -n rogi_xd -y python=3.9 && conda activate rogi_xd
    CUDA=cu116  # NOTE: depending on your machine, this can be any one of: ("cpu" | "cu116" | "cu117") 
    pip install torch==1.13 --extra-index-url https://download.pytorch.org/whl/${CUDA}\
    && pip install pyg-lib torch-scatter torch-sparse \
        torch-cluster torch-spline-conv torch-geometric \
        -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html \
    && pip install git+https://github.com/DeepGraphLearning/torchdrug \
    && pip install -r requirements.txt
    ```

3. install the `rogi_xd` pacakge: `pip install -e . --no-deps`


## Pretrained models

- to use the same pretrained models, run the following commands:
  ```bash
  git lfs install
  git lfs pull
  ```
  there should be two new directories: `models/gin/zinc` and `models/vae/zinc`

- both the VAE and GIN were pretrained over 100 epochs on the ZINC 250k dataset using a learning rate of `3e-4` and early stopping on the validation loss

- to train your own models, run one of the following commands:
  ```bash
  rogi_xd train -m (gin | vae) -d zinc -c 8
  ```
  The models will be saved to the following directory `models/{gin,vae}/zinc`, which can be supplied to the `rogi` command later via the `--model-dir` argument.

  _NOTE_: this script trains a simple GIN or VAE and _doesn't_ allow for custom architectures to specified. That's because the goal of this repository **was not** to provide _another_ VAE implementation. If you wish to reuse the VAE object model, then you'll want to head to the [`autoencoders` submodule](https://github.com/davidegraff/autoencoders)


## Results

All results can be generated via the following command: **`make all`**

### ROGI data

Use the `rogi_xd rogi` command line entry point to run your desired calculations.
```
usage: rogi_xd rogi [-h] [--logfile [LOGFILE]] [-v] (-i INPUT | -d DATASETS_TASKS [DATASETS_TASKS ...]) [-f {descriptor,morgan,chemberta,chemgpt,gin,vae,random}] [-r REPEATS] [-N N] [-o OUTPUT] [-b BATCH_SIZE]
                    [-m MODEL_DIR] [-c NUM_WORKERS] [--coarse-grain] [-k [NUM_FOLDS]] [--reinit] [--orig] [-l [LENGTH]]

optional arguments:
  -h, --help            show this help message and exit
  --logfile [LOGFILE], --log [LOGFILE]
                        the path to which the log file should be written. Not specifying will this log to stdout. Adding just the flag ('--log/--logfile') will automatically log to a file at 'logs/YYYY-MM-
                        DDTHH:MM:SS.log'
  -v, --verbose         the verbosity level
  -i INPUT, --input INPUT
                        A plaintext file containing a dataset/task entry on each line. Mutually exclusive with the '--datasets-tasks' argument
  -d DATASETS_TASKS [DATASETS_TASKS ...], --datasets-tasks DATASETS_TASKS [DATASETS_TASKS ...], --dt DATASETS_TASKS [DATASETS_TASKS ...], --datasets DATASETS_TASKS [DATASETS_TASKS ...]
  -f {descriptor,morgan,chemberta,chemgpt,gin,vae,random}, --featurizer {descriptor,morgan,chemberta,chemgpt,gin,vae,random}
  -r REPEATS, --repeats REPEATS
  -N N                  the number of data to subsample
  -o OUTPUT, --output OUTPUT
                        the to which results should be written. If unspecified, will write to 'results/raw/rogi/FEATURIZER.{csv,json}', depending on the output data ('.json' if '--cg' is present, '.csv' otherwise)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size
  -m MODEL_DIR, --model-dir MODEL_DIR
                        the directory of a saved model for VAE or GIN featurizers
  -c NUM_WORKERS, --num-workers NUM_WORKERS
                        the number of CPUs to parallelize data loading over, if possible
  --coarse-grain, --cg  whether to store the raw coarse-graining results.
  -k [NUM_FOLDS], --num-folds [NUM_FOLDS], --cv [NUM_FOLDS]
                        the number of folds to use in cross-validation. If this flag is present, then this script will run in cross-validation mode, otherwise it will just perform ROGI calculation. Adding only the flag
                        (i.e., just '-k') corresponds to a default of 5 folds, but a specific number may be specified
  --reinit              randomize the weights of a pretrained model before using it
  --orig                whether to use the original ROGI formulation (i.e., distance threshold as the x-axis). By default, uses the ROGI-XD formulation (i.e., 1 - log N_clusters / log N as the x-axis)
  -l [LENGTH], --length [LENGTH]
                        the length of a random representation
```

### Cross-validation and coarse-graining results

Use the same entrypoint as before with the addition of the `--cv` and `--cg` flags, like so: `rogi_xd rogi --cv --cg`

_Note_: The scripts rely datasets from both TDC [[1]] and GuacaMol oracle functions evaluated for random molecules sampled from ZINC250k [[2]]. The script will first search for the corresponding dataset in the `$ROGIXD_CACHE` directory (where `ROGIXD_CACHE` is an environment variable) and if it doesn't find them, will then download them to that directory. If this environment variable is not set, the scripts will use `$HOME/.cache/rogi_xd` instead.

[1]: https://tdcommons.ai/single_pred_tasks/overview/
[2]: https://tdcommons.ai/generation_tasks/molgen/


## Figures

See the corresponding notebook:
- [`correlation.ipynb`](./notebooks/correlation.ipynb): correlation plots and `$r` distribution plots
- [`rogi-dist.ipynb`](./notebooks/rogi-dist.ipynb): ROGI distribution boxplots and parity plots
- [`toy_surfaces.ipynb`](./notebooks/toy_surfaces.ipynb): toy example figures

_Note_: for these notebooks to work out of the box, data should be generated using the `make all` command from above`