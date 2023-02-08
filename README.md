# Pretrained Chemical Model Roughness

## Installation

### Conda
```
conda env create -f environment.yml
```
_Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu` before running the above command
```bash
DEVICE="cpu"
sed -i -e 's/cu116/cpu/g' environment.yml
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

  ```bash
  pcmr train -m (gin | vae) -d zinc -c 8
  ```
  The models will be saved to the following directory `models/{VAE,GIN}/zinc`, which can be supplied to the `rogi` command later via the `--model-dir` argument.

  _NOTE_: this script trains a simple GIN or VAE and _doesn't_ allow for custom architectures to specified. That's because the goal of this repository **was not** to provide _another_ VAE implementation. If you like the composable object model we used, feel free use it in your own project. I don't think a full citation is necessary, but a docstring reference and a shoutout would be appreciated :hug:.The following modules will contain most of the code you need:

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

```
./scripts/all.sh
```

### Figures

See the [`plots.ipynb`](./plots.ipynb) notebook