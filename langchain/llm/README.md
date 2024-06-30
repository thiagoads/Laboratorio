# LLMs c/ Langchain, FastAPI e LangServe

## Ambiente

- debian12 (bookworm)
- miniconda
- vscode (IDE) + jupyter notebook (extensão)
- OpenAI, LangChain Smith e GCP accounts

## Configuração

1) criar ambiente conda
   ```
   conda env create -f environment.yml
   conda activate langchain_llm
   ```

2) instalar pacotes restantes c/ pip
   ```
   pip install -qU langchain-google-vertexai langserve[all]
   ```

3) definir as váriáveis de ambiente:
    ```
    export LANGCHAIN_API_KEY=<YOUR_API_KEY>
    export OPENAI_API_KEY=<YOUR_API_KEY>
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials
    ```

## Execução

1) executar arquivo notebook.ipynb
   ```
   jupyter notebook
   ```

2) subir API na porta 8000 (certifique que não está em uso):
   ```
   python serve.py
   ```

3) acessar [LangServe Playground](http://localhost:8000/chain/playground/)



