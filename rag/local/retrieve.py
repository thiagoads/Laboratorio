import json

from llama_cpp import Llama

from libs.prompting import construct_prompt, stream_and_buffer
from libs.persistence import load_vector_store


if __name__ == '__main__':

    with open('config.json', mode='r') as file:
        config = json.load(file)
    
    LLM_PATH = config['llm_path']
    CHROMADB_PATH = config['chromadb_path']
    SYSTEM_PROMPT_FILE_PATH = config['prompts']["system_prompt"]
    USER_QUERY_FILE_PATH = config['prompts']["user_query"]

    # Loading documents from a vector database
    collection = load_vector_store(path=CHROMADB_PATH)
    
    # Retrieve documents according to similarity with user_query
    with open(USER_QUERY_FILE_PATH, 'r') as f:
        user_query = f.read()
        
    result = collection.query(query_texts=user_query, n_results=3)

    retrieved_docs = ''
    for document in result.get('documents'):
        for text in document:
            retrieved_docs = retrieved_docs + ' ' + text

    # Prompt question to llm based on retrieved context
    with open(SYSTEM_PROMPT_FILE_PATH, 'r') as f:
        system_prompt = f.read()

    # Build prompt using template
    prompt = construct_prompt(system_prompt=system_prompt, 
                              retrieved_docs=retrieved_docs, 
                              user_query=user_query)

    # TODO entender pq verbose=False dá erro LookupError: unknown encoding: ascii LLamaContext
    # https://github.com/abetlen/llama-cpp-python/issues/946
    llm = Llama(model_path=LLM_PATH, n_gpu_layers=1)
    stream_and_buffer(base_prompt=prompt, llm=llm)

    
        
    
