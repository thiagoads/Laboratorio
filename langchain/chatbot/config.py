import os

def check_environment_variables():
    is_ok = True
    if os.getenv("OPENAI_API_KEY") == None:
        print("OPENAI_API_KEY não configurada!")
        is_ok = False

    if os.getenv("LANGCHAIN_TRACING_V2") == None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if os.getenv("LANGCHAIN_API_KEY") == None:
        print("LANGCHAIN_API_KEY não configurada!")
        is_ok = False

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == None:
        print("GOOGLE_APPLICATION_CREDENTIALS não configurada!")
        is_ok = False

    return is_ok