# Vector Database c/ Chroma

## Ambiente

- debian12 (bookworm)
- miniconda
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) criar container docker
   ```
   ./run.sh
   ```
2) criar ambiente conda
   ```
   conda env create -f environment.yml
   conda activate chroma
   ```

## Execução

1) executar arquivo notebook.ipynb
   ```
   jupyter notebook
   ```