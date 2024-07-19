import requests
import json
from utils import stream_print


'''

'''


while True:
    question = input("请问关于《大模型赋能数据库调优》这个实验，您有什么问题？可以随时想我提问哦\n\n")
    data = {'q':question}
    data = json.dumps(data)

    response = requests.post('http://127.0.0.1:8080/db_tuner_QA', data=data, stream=True)

    print('\n')
    for line in response.iter_lines():
        if line:
            d = json.loads(line)
            stream_print(d['res'])
    print('\n')
    stream_print("======"*15, 0.001)
    print('\n')