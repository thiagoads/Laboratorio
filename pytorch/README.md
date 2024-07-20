# Primeiros passos com Pytorch

Base on:
https://pytorch.org/get-started/locally/

## Ambiente

- debian12 (bookworm)
- miniconda
- pytorch
- vscode (IDE)
- jupyter notebook (extensão)
- TensorBoard (extensão)

## Configuração

1) criar ambiente conda
```
conda env create -f environment.yml
conda activate pytorch
```

2) Ou, alternativamente:

```
conda create -n pytorch python=3.9.16
```

CPU
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

GPU
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```


## Execução

1) overview.ipynb
2) tensors.ipynb
3) autograd.ipynb
4) models.ipynb
5) tensorboard.ipynb
6) training.ipynb
7) understanding.ipynb

