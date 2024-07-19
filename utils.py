from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
import json
import pandas as pd
import csv
import numpy as np
import time
import sys

API_KEY = 'sk-exC6cb6BemF27sRP618dC8A8Ff684d2f8bF4C6Ac46F8D757'
client = OpenAI(api_key=API_KEY, base_url="https://www.jcapikey.com/v1")


def get_chatgpt_response(query):
     sys_info = '请回答下列问题: {}. 答案:'
     prompt = sys_info.format(query)
     responses = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo"
    )
     
     return responses.choices[0].message.content


def stream_print(output, speed=0.04):
    for i in output:
        sys.stdout.write(i)
        sys.stdout.flush()
        time.sleep(speed)


def get_embedding(text_to_embed, model="text-embedding-3-small"):
	# Embed a line of text
	response = client.embeddings.create(
    	model= model,
    	input=[text_to_embed]
	)
	# 将 AI 输出嵌入提取为浮点数列表
	embedding = response.data[0].embedding
    
	return embedding


def cosine_similarity(a, b):
    if isinstance(a, str):
         a = json.loads(a)
    # a = np.asarray(a, dtype='float64')
    # b = np.asarray(b, dtype='float64')
    return dot(a, b)/(norm(a)*norm(b))


def search_reviews(config, product_description, n=1):
   embed_path = config['qa_data_with_embedding'] 
   df = pd.read_csv(embed_path)
   embedding = get_embedding(product_description, model='text-embedding-3-small')
   df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
   res = df.sort_values('similarities', ascending=False).head(n)
   return res


def convert_json_to_csv(input_json, output_csv):
    knowledge = None
    with open(input_json, 'r') as f:
        knowledge = json.load(f)
    
    keys = list(knowledge[0].keys())

    list_json_data = [keys]
    for i in range(len(knowledge)):
        list_json_data.append([str(knowledge[i][k]) for k in keys])

    d = pd.DataFrame(list_json_data)
    d.to_csv(output_csv, header=0, index=False)


def get_q_embedding(input_datapath, output_datapath, embedding_model="text-embedding-3-small"):
    df = pd.read_csv(input_datapath)
    df['embedding'] = df.q.apply(lambda x: get_embedding(x, model=embedding_model))
    df.to_csv(output_datapath, index=False)


def prepare_knowledge(config):
     input_datapath = config['og_qa_data_json_path'] 
     qa_data_csv_path = config['qa_data_csv_path']
     qa_data_with_embedding = config['qa_data_with_embedding']
     convert_json_to_csv(input_datapath, qa_data_csv_path)
     get_q_embedding(qa_data_csv_path, qa_data_with_embedding)


def load_config(path = './config.json'):
    config = None
    with open(path, 'r') as f:
        config = json.load(f)

    return config