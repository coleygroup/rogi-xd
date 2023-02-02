# Pretrained Chemical Model Roughness

## Installation

### Conda
_Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu`
```
conda env create -f environment.yml
```

### Manual
_Note_: this is only recommended in the event that you'd like specify different package versions
```sh
conda create -n pcmr -y python=3.9 && conda activate pcmr
CUDA=cu117  #NOTE: this can be: "cpu", "cu116", or "cu117" depending on your device
pip install torch==1.13 \
  && pip install pyg-lib torch-scatter torch-sparse \
    torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html \
  && pip install git+https://github.com/DeepGraphLearning/torchdrug \
  && pip install .
```


## Training models

- VAE was trained for 100 epochs using Adam and LR = 3e-4 
-