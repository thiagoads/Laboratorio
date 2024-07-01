from operator import itemgetter

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_message_histories import ChatMessageHistory

from config import check_environment_variables
from models import choose_model_instance
from utils import get_trimmer, prompt_with_language, prompt

store = {}
parser = StrOutputParser()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def quick_start(model: BaseChatModel):

    chain = model | parser
    message = chain.invoke([HumanMessage(content="Hi! I'm Thiago")])
    print(message)

    message = chain.invoke([HumanMessage(content="What's my name?")])
    print(message)

    message = chain.invoke(
        [
                HumanMessage(content="Hi! I'm Thiago"),
                AIMessage(content="Hello Thiago! How can I assist you today?"),
                HumanMessage(content="What's my name?"),
        ]
    )
    print(message)

# TODO aqui está dando erro com o vertex ai
# limitando o tamanho do contexto vindo do histórico
def managing_conversation_history(model: BaseChatModel):
    
    trimmer = get_trimmer(model)

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm thiago"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    chain = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt_with_language | model | parser

    # aqui o model não vai se lembrar pois não está no histórico
    message = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what's my name?")],
            "language": "English",
        }
    )

    print(message)

    # aqui o model se lembra pois está no histórico
    message = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what math problem did i ask?")],
            "language": "English",
        }
    )

    print(message)

    # add histório no chain
    with_message_history = RunnableWithMessageHistory(chain, get_session_history,input_messages_key="messages",)

    config = {"configurable": {"session_id": "abc11"}} # TODO checar essa session

    response = with_message_history.invoke(
        {
            "messages": [HumanMessage(content="what is my name?")],
            "language": "English",
        },
        config=config,
    )

    print(response)

# trabalhando com prompt templates
def prompt_templates(model: BaseChatModel):
    chain = prompt | model | parser

    message = chain.invoke({"messages": [HumanMessage(content="Hi! I'm Thiago")]})

    print(message)    

    # a partir daqui só o da openai funciona como esperado
    # o chat do vertex ai não response na linguagem informada
    chain = prompt_with_language | model | parser

    message = chain.invoke({"messages": [HumanMessage(content="Hi! I'm Thiago")], "language": "Spanish"})
    print(message)

    with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")
    config = {"configurable": {"session_id": "abc11"}}
    
    message = with_message_history.invoke(
        {"messages": [HumanMessage(content="Hi! I'm Thiago")], "language": "Portuguese"}, config=config
    )
    print(message)

    message = with_message_history.invoke(
        {"messages": [HumanMessage(content="What's my name?")], "language": "Portuguese"}, config=config
    )
    print(message)

    # testando passar question em portugues para ver o que acontece
    message = with_message_history.invoke(
        {"messages": [HumanMessage(content="Como eu me chamo?")], "language": "Portuguese"}, config=config
    )
    print(message)


# recuperando mensagem do histórico de conversação
def keep_tracking_conversation(model: BaseChatModel):
    with_message_history = RunnableWithMessageHistory(model, get_session_history)
    config = {"configurable": {"session_id": "abc2"}}
    chain = with_message_history | parser
    message = chain.invoke([HumanMessage(content = "Hi! I'm Thiago")], config=config)
    print(message)

    config = {"configurable": {"session_id": "abc3"}}
    message = chain.invoke([HumanMessage(content = "What's my name?")], config=config)
    print(message)

    config = {"configurable": {"session_id": "abc2"}}
    message = chain.invoke([HumanMessage(content = "What's my name?")], config=config)
    print(message)



# trabalhando com streamings para melhorar ux
def streaming_responses(model: BaseChatModel):
    
    trimmer = get_trimmer(model)

    chain = RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) | prompt_with_language | model | parser

    with_message_history = RunnableWithMessageHistory(chain, get_session_history,input_messages_key="messages",)

    config = {"configurable": {"session_id": "abc15"}}

    for r in with_message_history.stream(
        {
            "messages": [HumanMessage(content="hi! I'm thiago. tell me a joke")],
            "language": "English",
        },
        config=config
    ): print(r, end="|")

    print()


if __name__ == "__main__":

    if check_environment_variables():
        model = choose_model_instance()

        print("#-------------------------------------#")
        print('# Iniciando uma conversa básica       #')
        print("#-------------------------------------#")
        quick_start(model)

        print("#-------------------------------------#")
        print('# Mantendo histórico da conversa      #')
        print("#-------------------------------------#")
        keep_tracking_conversation(model)

        print("#-------------------------------------#")
        print('# Trabalhando com prompt templates    #')
        print("#-------------------------------------#")
        prompt_templates(model)
        
        print("#-------------------------------------#")
        print("# Gerenciando o histórico da conversa #")
        print("#-------------------------------------#")
        managing_conversation_history(model)

        print("#-------------------------------------#")
        print("# Melhorando UX no chat com streaming #")
        print("#-------------------------------------#")
        streaming_responses(model)
    else:
        exit("Existem variáveis de ambiente não configuradas!")