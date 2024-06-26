# Usando MLFlow c/ tensorflow

## Ambiente

- debian12 (bookworm)
- miniconda
- MLFlow
- VSCode (IDE) + jupyter notebook (extensão)
- Postman ou curl

## Configuração

1) instalar MLFlow como tracking server


2) criar ambiente conda
    ```
    conda env create -f environment.yml
    conda activate tensorflow_mlflow
    ```

3) importe collection no Postman
    ```
    Import >> postman.json
    ```

## Execução

1) abrir o notebook notebook.ipynb no vscode

2) selecione o kernel tensorflow_mlflow

3) execute o código do notebook

4) acompanhe o resultado do treinamento em:

    > http://127.0.0.1:5000
