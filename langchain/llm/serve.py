from typing import List

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI

system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()
model = ChatVertexAI(model = "gemini-pro")

chain = prompt_template | model | parser

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="...",
)


add_routes(app, chain, path="/chain")


if __name__ == "__main__":
    import uvicorn

    # se já tiver em uso, pode mudar a porta aqui
    uvicorn.run(app, host="localhost", port=8000)
