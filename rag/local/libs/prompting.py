from llama_cpp import Llama
import sys

def stream_and_buffer(base_prompt, llm, max_tokens=800, stop=['Q:', '\n'], echo=True, stream=True):
    
    formated_prompt = f'Q: {base_prompt} A: '

    response = llm(formated_prompt, max_tokens=max_tokens, stop=stop, echo=echo, stream=stream)

    buffer = ''

    for message in response:
        chunk = message['choices'][0]['text']
        #print('Chunk -> ', chunk, ' Len -> ', len(chunk))
        buffer += chunk.replace("'", '*')
        words = buffer.split(' ')

        for word in words[:-1]:
            sys.stdout.write(word + ' ')
            sys.stdout.flush()

        buffer = words[-1]

    if buffer:
        sys.stdout.write(buffer)    
        sys.stdout.flush()
    
def construct_prompt(system_prompt, retrieved_docs, user_query):
    print('Building prompts from template.')
    prompt = f"""{system_prompt}

    Here is the retrieved context:
    {retrieved_docs}

    Here is the user query:
    {user_query}
    """
    print('Prompt ->', prompt)
    return prompt


