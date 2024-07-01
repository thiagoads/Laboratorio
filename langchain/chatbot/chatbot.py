from operator import itemgetter

from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_message_histories import ChatMessageHistory

from config import check_environment_variables
from models import choose_model_instance
#from conversation import start_conversation
from utils import get_trimmer, prompt

store = {}
parser = StrOutputParser()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def start_conversation(model:BaseChatModel):

    input_question = "Question: "
    
    trimmer = get_trimmer(model)

    chain = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt | model | parser

    with_message_history = RunnableWithMessageHistory(chain, get_session_history,input_messages_key="messages",)

    import uuid

    config = {"configurable": {"session_id": uuid.uuid4()}}

    question = input(input_question)

    while(question):
        answer = with_message_history.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        print(f"Answer: {answer}")
        question = input(input_question)


if __name__ == "__main__":

    if check_environment_variables():
        model = choose_model_instance()

        print("#-------------------------------------#")
        print('# Iniciando uma conversa c/ chatbot   #')
        print("#-------------------------------------#")
        start_conversation(model)

    else:
        exit("Existem variáveis de ambiente não configuradas!")