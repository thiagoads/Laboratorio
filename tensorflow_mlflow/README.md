# Usando MLFlow c/ tensorflow

## Ambiente

- debian12 (bookworm)
- Docker
- miniconda
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) criar container MLFlow

    a) na pasta mlflow, execute ./run.sh

    b) ou, ./build.sh e ./create.sh

    c) use ./destroy p/ remover o conteiner

2) criar ambiente conda
    ```
    conda env create -f environment.yml
    conda activate tensorflow_mlflow
    ```

## Execução

1) abrir o notebook notebook.ipynb no vscode

2) selecione o kernel tensorflow_mlflow

3) execute o código do notebook

4) acompanhe o resultado do treinamento em:

    > http://127.0.0.1:5000
