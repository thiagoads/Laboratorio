import chromadb


def create_vector_store(documents, path, embedding_function):
    print('Persisting data in vector database.')
    chroma = chromadb.PersistentClient(path=path)

    collection = chroma.get_or_create_collection(name="rag_local", embedding_function=embedding_function)

    for _, chunks in documents.items():
        for chunk_id, chunk_dict in chunks.items():
            text = chunk_dict.get('text')
            metadata = chunk_dict.get('metadata')
            collection.add(documents=text, metadatas=metadata, ids=chunk_id)
    return collection

def load_vector_store(path):
    print('Persisting data in vector database.')
    chroma = chromadb.PersistentClient(path=path)
    return chroma.get_or_create_collection(name="rag_local")