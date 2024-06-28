# Implementando RAG

## Ambiente

- debian12 (bookworm)
- miniconda
- chromadb
- langchain 
- chatgpt api
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) criar ambiente conda
```
conda env create -f environment.yml
conda activate rag_basic
```

2) instale o pacote abaixo com pip
```
pip install langchain-community.py
```

3) configure sua API KEY da OpenAI
```
export set OPENAI_API_KEY=<YOUR_API_KEY>
```

## Execução

1) abrir o notebook notebook.ipynb no vscode
   ```
   code .
   ```

2) selecione o kernel rag_basic
