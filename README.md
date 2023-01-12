# Pretrained Chemical Model Roughness

## Installation

### Conda
_Note_: `environment.yml` contains CUDA 11.6 versions of several packages. If you want to use this environment on CPU, replace `cu116` with `cpu`
```
conda env create -f environment.yml
```

### Manual
_Note_: this is only recommended in the event that you'd like specify different package versions
```
conda create -n pcmr python=3.9
pip install torch==1.13
CUDA=cu117 # NOTE: this can be: "cpu", "cu116", or "cu117" depending on your CUDA version
pip install torch-scatter torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+${CUDA}.html
pip install git+git+https://github.com/DeepGraphLearning/torchdrug
pip install gt4sd
```
