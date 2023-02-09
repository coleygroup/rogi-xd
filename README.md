# Pretrained Chemical Model Roughness

## Setup

### Conda
```
conda env create -f environment.yml
```
_Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu` before running the above command
```bash
sed -i -e 's/cu116/cpu/g' environment.yml
conda env create -f environment.yml
```
### Manual
_Note_: this is only recommended in the event that you'd like specify different package versions
```sh
conda create -n pcmr -y python=3.9 && conda activate pcmr
CUDA=cu116  # NOTE: depending on your machine, this can be any one of: ("cpu" | "cu116" | "cu117") 
pip install torch==1.13 --extra-index-url https://download.pytorch.org/whl/${CUDA}\
  && pip install pyg-lib torch-scatter torch-sparse \
    torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html \
  && pip install git+https://github.com/DeepGraphLearning/torchdrug \
  && pip install .
```


## Training models

- both the VAE and GIN were trained over 100 epochs on the ZINC 250k dataset using a learning rate of `3e-4` and early stopping on the validation loss
- to train your own models, run the following command:

  ```
  pcmr train -m (gin | vae) -d zinc -c 8
  ```
  The models will be saved to the following directory `models/{GIN,VAE}/zinc`, which can be supplied to the `rogi` command later via the `--model-dir` argument.

  _NOTE_: this script trains a simple GIN or VAE and _doesn't_ allow for custom architectures to specified. That's because the goal of this repository **was not** to provide _another_ VAE implementation. If you like the composable object model we used, feel free use it in your own project. I don't think a full citation is necessary, but a docstring reference and a shoutout would be appreciated :hugs:.The following modules will contain most of the code you need:

  - `pcmr.models.vae`
  - `pcmr.models.gin`
  - `pcmr.cli.train` 

- to use the same pretrained models, run the following commands:
```bash
git lfs install
git lfs pull
```
there should be two new directories: `models/GIN/zinc` and `models/VAE/zinc`

## Reproducing results

### Raw data

Use the `pcmr rogi` command line entry point to run your desired calculations. All results can be generated using the [`scripts/all.sh`](./scripts/all.sh) script.
```
$ pcmr rogi --help
usage: pcmr rogi [-h] [--logfile [LOGFILE]] [-v] (-i INPUT | -d DATASETS_TASKS [DATASETS_TASKS ...]) [-f {descriptor,chemberta,chemgpt,gin,vae}] [-r REPEATS] [-N N] [-o OUTPUT] [-m MODEL_DIR] [-b BATCH_SIZE]
                 [-c NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --logfile [LOGFILE], --log [LOGFILE]
                        the path to which the log file should be written. Not specifying will this log to stdout. Adding just the flag ('--log/--logfile') will automatically log a file at 'logs/YYYY-MM-DDTHH:MM:SS.log'
  -v, --verbose         the verbosity level
  -i INPUT, --input INPUT
                        A plaintext file containing a dataset/task entry on each line. Mutually exclusive with the '--datasets-tasks' argument
  -d DATASETS_TASKS [DATASETS_TASKS ...], --datasets-tasks DATASETS_TASKS [DATASETS_TASKS ...], --dt DATASETS_TASKS [DATASETS_TASKS ...], --datasets DATASETS_TASKS [DATASETS_TASKS ...]
  -f {descriptor,chemberta,chemgpt,gin,vae}, --featurizer {descriptor,chemberta,chemgpt,gin,vae}
  -r REPEATS, --repeats REPEATS
  -N N                  the number of data to sumbsample
  -o OUTPUT, --output OUTPUT
                        the to which results should be written. If unspecified, will write to 'results/raw/FEATURIZER.csv'
  -m MODEL_DIR, --model-dir MODEL_DIR
                        the directory of a saved model for VAE or GIN featurizers
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the batch size to use in the featurizer. If unspecified, the featurizer will select its own batch size
  -c NUM_WORKERS, --num-workers NUM_WORKERS
                        the number of CPUs to parallelize data loading over, if possible.
```

_Note_: We rely datasets from both the TDC [[1],[2]] and DOCKSTRING [[3]]. The script will search for the corresponding dataset in the `$PCMR_CACHE` directory (where `PCMR_CACHE` is an environment variable) and if it doesn't find them will download them to that directory. If this environment variable is not set, the scripts will automatically download the files to `~/.cache/pcmr` (where `~` is your home directory).

[1]: https://tdcommons.ai/single_pred_tasks/overview/
[2]: https://tdcommons.ai/generation_tasks/molgen/
[3]: https://figshare.com/articles/dataset/dockstring_dataset/16511577?file=35948138

### Figures

See the [`plots.ipynb`](./plots.ipynb) notebook