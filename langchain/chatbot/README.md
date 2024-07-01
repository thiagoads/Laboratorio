# Aplicação de LLM c/ Langchain, FastAPI e LangServe

## Ambiente

- debian12 (bookworm)
- miniconda
- vscode (IDE) + jupyter notebook (extensão)
- LangChain Smith, OpenAI and GCP accounts

## Configuração

1) criar ambiente conda
   ```
   conda env create -f environment.yml
   conda activate langchain_chatbot
   ```

2) instalar pacotes restantes c/ pip
   ```
   pip install -qU langchain-google-vertexai
   ```

3) definir as váriáveis de ambiente:
    ```
    export LANGCHAIN_API_KEY=<YOUR_API_KEY>
    export OPENAI_API_KEY=<YOUR_API_KEY>
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials
    ```

## Execução

1) execute o arquivo tutorial.py
   ```
   python tutorial.py
   ```

1) execute o arquivo chatbot.py
   ```
   python chatbot.py
   ```

