# Local RAG
Based on: https://towardsdatascience.com/local-rag-from-scratch-3afc6d3dea08

## Ambiente

- debian12 (bookworm)
- miniconda
- chromadb
- llama_cpp
- transformers
- pytorch
- vscode (IDE) + jupyter notebook (extensão)

## Configuração

1) criar ambiente conda
```
conda env create -f environment.yml
conda activate rag_local
```

2) atualize dependencias de transformers
```
pip install transformers -U
```

3) download llm model localmente
```
wget https://huggingface.co/janhq/mistral/resolve/f8a13e8d0b76e31afc9652d814ce1ec658a06ee7/mistral-7b-instruct-v0.2.Q3_K_L.gguf
```

4) configurar caminho local p/ llm em config.json 

## Execução

1) salvar documentos invidualmente
   ```
   python collect.py
   ```

2) indexar documentos no vectordb
   ```
   python index.py
   ```

3) perguntar ao llm baseado no em contexto
   ```
   python retrieve.py
   ```

4) ou, alternativamente, executar tudo
   ```
   ./run.sh
   ```

Obs: Altere os prompts nos arquivos system_prompt.txt e user_query.txt na pasta prompts.

## Referências

https://huggingface.co/mistralai