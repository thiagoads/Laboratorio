import json

from libs.chunking import document_chunker
from libs.caching import download_cache_model_locally, is_model_cached_locally, load_model_from_cache
from libs.persistence import create_vector_store
from libs.embeddings import MyEmbeddingFunction


if __name__ == '__main__':

    with open('config.json', mode='r') as file:
        config = json.load(file)
    
    MODEL_NAME = config['model_name']
    CACHE_DIR_PATH = config['cache_dir_path']
    DOCUMENTS_PATH = config['documents_path']
    CHROMADB_PATH = config['chromadb_path']

    if not is_model_cached_locally(model_name=MODEL_NAME, 
                                   model_cache_path=CACHE_DIR_PATH):
        # Download and cache model locally if it's not present
        download_cache_model_locally(model_name=MODEL_NAME, 
                                     model_cache_path=CACHE_DIR_PATH)

    # Load model from cache
    tokenizer, model = load_model_from_cache(model_name=MODEL_NAME, model_cache_path=CACHE_DIR_PATH)


    # Chunking data
    documents = document_chunker(directory_path=DOCUMENTS_PATH, 
                                 model_name=MODEL_NAME, 
                                 chunk_size=256)


    # Persisting documents in a vector database using custom embedding function
    collection = create_vector_store(documents=documents,
                                     path=CHROMADB_PATH,
                                     embedding_function=MyEmbeddingFunction(model=model, tokenizer=tokenizer))
    
        
    
