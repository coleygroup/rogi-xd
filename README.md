# Pretrained Chemical Model Roughness

## Setup

As a first step, clone this repository and its submodules:
```
git clone --recurse-submodules THIS_REPO
```

### Conda
```
conda env create -f environment.yml
```
_Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu` before running the above command
```
sed -i -e 's/cu116/cpu/g' environment.yml
conda env create -f environment.yml
```
### Manual
_Note_: this is only recommended in the event that you'd like specify different package versions
```
conda create -n pcmr -y python=3.9 && conda activate pcmr
CUDA=cu116  # NOTE: depending on your machine, this can be any one of: ("cpu" | "cu116" | "cu117") 
pip install torch==1.13 --extra-index-url https://download.pytorch.org/whl/${CUDA}\
  && pip install pyg-lib torch-scatter torch-sparse \
    torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html \
  && pip install git+https://github.com/DeepGraphLearning/torchdrug \
  && pip install . autoencoders
```


## Pretrained models

- both the VAE and GIN were pretrained over 100 epochs on the ZINC 250k dataset using a learning rate of `3e-4` and early stopping on the validation loss
- to train your own models, run the following command:

  ```bash
  pcmr train -m (gin | vae) -d zinc -c 8
  ```
  The models will be saved to the following directory `models/{gin,vae}/zinc`, which can be supplied to the `rogi` command later via the `--model-dir` argument.

  _NOTE_: this script trains a simple GIN or VAE and _doesn't_ allow for custom architectures to specified. That's because the goal of this repository **was not** to provide _another_ VAE implementation. If you like the composable object model we used, feel free use it in your own project. I don't think a full citation is necessary, but a docstring reference and a shoutout would be appreciated :hugs:.The following modules will contain most of the code you need:

  - `pcmr.models.gin`
  - `pcmr.cli.train` 

- to use the same pretrained models, run the following commands:

  ```bash
  git lfs install
  git lfs pull
  ```

  there should be two new directories: `models/gin/zinc` and `models/vae/zinc`


## Results

All results can be generated via the following command: `make all`

### ROGI data

Use the `pcmr rogi` command line entry point to run your desired calculations.

```
$ usage: pcmr rogi [-h] [--logfile [LOGFILE]] [-v] (-i INPUT | -d DATASETS_TASKS [DATASETS_TASKS ...]) [-f {descriptor,chemberta,chemgpt,gin,vae}] [-r REPEATS] [-N N] [-o OUTPUT] [-b BATCH_SIZE] [-m MODEL_DIR] [-c NUM_WORKERS] [--coarse-grain] [-k [NUM_FOLDS]] [--reinit]

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
  --v1                  whether to use the v1 ROGI formulation (distance threshold as the x-axis). By default, uses v2 (1 - log N_clusters / log N as the x-axis)
```

### Cross-validation and coarse-graining results

Use the same entrypoint as before with the addition of the `--cv` and `--cg` flags, like so: `pcmr rogi --cv --cg`

_Note_: The scripts rely datasets from both TDC [[1],[2]]. The script will first search for the corresponding dataset in the `$PCMR_CACHE` directory (where `PCMR_CACHE` is an environment variable) and if it doesn't find them, will then download them to that directory. If this environment variable is not set, the scripts will use `$HOME/.cache/pcmr` instead.

[1]: https://tdcommons.ai/single_pred_tasks/overview/
[2]: https://tdcommons.ai/generation_tasks/molgen/
[3]: https://figshare.com/articles/dataset/dockstring_dataset/16511577?file=35948138


## Figures

See the corresponding notebook:
- [`auc.ipynb`](./notebooks/corrleation.ipynb): loss of dispersion plots
- [`correlation.ipynb`](./notebooks/correlation.ipynb): correlation plots and `$r` distribution plots
- [`rogi-dist.ipynb`](./notebooks/rogi-dist.ipynb): ROGI distribution boxplots and parity plots
- [`toy_surfaces.ipynb`](./notebooks/toy_surfaces.ipynb): toy example figures

_Note_: for these notebooks to work out of the box, data should be generated and organized using the shell scripts above