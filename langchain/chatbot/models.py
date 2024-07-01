from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI

_MODEL_OPTION_MESSAGE = "[1] OpenAI(default) [2] VertexAI"
_MODEL_OPTION_DEFAULT = ChatOpenAI()
_MODEL_OPTION_OPENAI = "1"
_MODEL_OPTION_VERTEXAI = "2"


def choose_model_instance() -> BaseChatModel:
    print("Modelos: {}".format(_MODEL_OPTION_MESSAGE))
    option = input("Seleção: ")
    model = _MODEL_OPTION_DEFAULT
    if option == _MODEL_OPTION_OPENAI:
        model = ChatOpenAI()
    elif option == _MODEL_OPTION_VERTEXAI:
        model = ChatVertexAI()
    elif option != None and len(option) > 0:
        exit('Opção inválida!')
    return model

