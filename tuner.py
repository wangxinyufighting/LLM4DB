
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import json
from openai import OpenAI
from utils import search_reviews, get_chatgpt_response, load_config
import pandas as pd
import numpy as np

app = FastAPI()

class question(BaseModel):
    q: str = None

@app.post('/db_tuner_QA')
def calculate(request_data: question):
    q = request_data.q
    knowledge = None
    config = load_config()
    
    with open('./data/qa.json', 'r') as f:
        knowledge = json.load(f)

    res = {"res":None}
    for qa in knowledge:
        if qa['q'] == q:
            res["res"] = qa['a']
            return res
        
        sim_q_res = search_reviews(config, q, n=1)
        if list(sim_q_res['similarities'])[0] > 0.5:
            res['res'] = list(sim_q_res['a'])[0]
            return res
    
    if res['res'] is None:
       response = get_chatgpt_response(q)
       res["res"] = "对不起同学，这个问题似乎和本实验无关哦~ \n\n 不过我可以尝试回答一下 \n\n{} \n\n 如果没有帮到你，你可以访问 https://github.com/SolidLao/GPTuner 仔细查看实验步骤哦~".format(response)

    return res


if __name__ == '__main__':
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8080,
                workers=1)
