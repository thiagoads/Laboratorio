import requests
import os

model_id = 'facebook/bart-large-cnn'

API_TOKEN = os.environ.get('HUGGING_FACE_API_TOKEN')

API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

with open('doc.txt', encoding='utf-8') as f:
    text = f.read()

data = {
    'inputs' : text,
    'parameters': {'do_sample': False}
}

answer = query(data)


print('Original -> ', text)
print('')
print('Summary  ->', answer[0]['summary_text'])



