from transformers import AutoModel, AutoTokenizer
import torch


from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model:AutoModel, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inputs: Documents) -> Embeddings:
        tokens = self.tokenizer(inputs, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1).squeeze()
        return embeddings.tolist()

def compute_embeddings(text: str, model: AutoModel, tokenizer: AutoTokenizer) -> list:
    print('computing embeddings')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.tolist()

def create_vector_store(documents: dict, model:AutoModel, tokenizer:AutoTokenizer):
    print('creating vector store')
    vector_store = {}
    for doc_id, chunks in documents.items():
        doc_vectors = {}
        for chunk_id, chunk_dict in chunks.items():
            doc_vectors[chunk_id] = compute_embeddings(chunk_dict.get("text"), model=model, tokenizer=tokenizer)
        vector_store[doc_id] = doc_vectors
    return vector_store
