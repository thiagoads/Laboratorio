# Usando gpu c/ tensorflow

## Ambiente

- debian12 (bookworm)
- nvidia geforce mx150
- nvidia-driver
- cuda-toolkit
- miniconda
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) instalar driver nvidia e cuda, conforme instruções do arquivo:
```
nvidia-and-cuda.txt
```

2) abrir aplicativo NVIDIA Settings ao final do passo anterior

![NVIDIA Settings](/nvidia-settings.png)


3) criar ambiente conda
```
conda env create -f environment.yml
conda activate tensorflow_gpu
```

## Execução

1) abrir o notebook tensorflow_gpu.ipynb no vscode

2) selecione o kernel tensorflow_gpu

3) ou, alternativamente rode o arquivo script.py
```
python script.py
```
4) acompanhe o uso de gpu no aplicativo NVIDIA Settings
