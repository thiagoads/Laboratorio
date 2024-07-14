import os
from transformers import AutoModel, AutoTokenizer

def load_model_from_cache(model_name: str, model_cache_path: str):
    print('Loading model from cache folder.')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def download_cache_model_locally(model_name: str, model_cache_path: str):
    print('Download pretrained model.')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print('Saving model in local cache folder.')
    tokenizer.save_pretrained(model_cache_path + "/tokenizer")
    model.save_pretrained(model_cache_path + "/embeddings")

def is_model_cached_locally(model_name: str, model_cache_path: str) -> bool:
    print('Checking if model is present locally.')
    is_present = os.path.isdir(model_cache_path)
    is_present = is_present and os.path.isdir(model_cache_path + '/tokenizer')
    is_present = is_present and os.path.isdir(model_cache_path + '/embeddings')
    if is_present == False:
        print('Model wasnt found locally!')
    return is_present
