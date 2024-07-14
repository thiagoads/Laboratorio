import re
import os
import uuid
from transformers import AutoTokenizer


def document_chunker(directory_path, 
                      model_name, 
                      paragraph_separator='\n\n', 
                      chunk_size = 1024, # número de tokens em um chunk
                      separator=' ', 
                      secondary_chunking_regex=r'\S+?[\.,;!?]', 
                      chunk_overlap=0):
    print('Chunking documents')
    documents = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=False)

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        basename = os.path.basename(filepath)
        sku = os.path.splitext(basename)[0]
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
       
        doc_id = str(uuid.uuid4())

        paragraphs = re.split(paragraph_separator, text)
        all_chunks = {}

        for paragraph in paragraphs:
            words = paragraph.split(separator)
            current_chunk = ''
            chunks = []

            for word in words:
                new_chunk = current_chunk + (separator if current_chunk else '') + word
                if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                    current_chunk = new_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word

            if current_chunk:
                chunks.append(current_chunk)

            # TODO entender daqui pra baixo
            refined_chunks = []
            for chunk in chunks:
                if len(tokenizer.tokenize(chunk)) > chunk_size:
                    sub_chunks = re.split(secondary_chunking_regex, chunk)
                    sub_chunk_accum = ""
                    for sub_chunk in sub_chunks:
                        if sub_chunk_accum and len(tokenizer.tokenize(sub_chunk_accum + sub_chunk + ' ')) > chunk_size:
                            refined_chunks.append(sub_chunk_accum.strip())
                            sub_chunk_accum = sub_chunk
                        else:
                            sub_chunk_accum += (sub_chunk + ' ')
                    if sub_chunk_accum:
                        refined_chunks.append(sub_chunk_accum.strip())
                else:
                    refined_chunks.append(chunk)

            final_chunks = []
            if chunk_overlap > 0 and len(refined_chunks) > 1:
                for i in range(len(refined_chunks) - 1):
                    final_chunks.append(refined_chunks[i])
                    overlap_start = max(0, len(refined_chunks[i]) - chunk_overlap)
                    overlap_end = min(chunk_overlap, len(refined_chunks[i+1]))
                    overlap_chunk = refined_chunks[i][overlap_start:] + ' ' + refined_chunks[i+1][:overlap_end]
                    final_chunks.append(overlap_chunk)
                final_chunks.append(refined_chunks[-1])
            else:
                final_chunks = refined_chunks

            # Assign a UUID for each chunk and structure it with text and metadata
            for chunk in final_chunks:
                chunk_id = str(uuid.uuid4())
                all_chunks[chunk_id] = {"text": chunk, "metadata": {"file_name":sku}}  # Initialize metadata as dict

        # Map the document UUID to its chunk dictionary
        documents[doc_id] = all_chunks


    return documents
