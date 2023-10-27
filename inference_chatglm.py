# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     inference_chatglm
   Author :       MSI-NB
   date：          2023/10/26
   Change Activity:
                   2023/10/26:
-------------------------------------------------
   功能：
-------------------------------------------------
"""

from transformers import pipeline, set_seed
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import os
import re, json, fire

from tqdm import tqdm
import random
import requests
import collections
from datetime import datetime

import zhipuai


def async_invoke_example():
    response = zhipuai.model_api.async_invoke(
        model="chatglm_pro",
        prompt=[
            {"role": "user", "content": "人工智能"},
            {"role": "user", "content": "外星生命"},
                ],
        top_p=0.7,
        temperature=0.9,
    )
    print(response)
    # response = zhipuai.model_api.query_async_invoke_result("516016983114764598059177926762529200")

def query_async_invoke_result_example():
    response = zhipuai.model_api.query_async_invoke_result("your task_id")
    print(response)

def invoke_example(inputs:str, model_name,top_p=0.7, temperature=0.9):
    response = zhipuai.model_api.invoke(
        model=model_name,
        prompt=[{"role": "user", "content": inputs}],
        top_p=top_p,
        temperature=temperature,
    )
    print(response)
    print(response["data"]["choices"][0]["content"])
    return response["data"]["choices"][0]["content"],response["data"]["usage"]["prompt_tokens"]

'''
  说明：
  add: 事件流开启
  error: 平台服务或者模型异常，响应的异常事件
  interrupted: 中断事件，例如：触发敏感词
  finish: 数据接收完毕，关闭事件流
'''
def sse_invoke_example(inputs:str, top_p=0.7, temperature=0.9):
    response = zhipuai.model_api.sse_invoke(
        model="chatglm_pro",
        prompt=[{"role": "user", "content": inputs}],
        top_p=top_p,
        temperature=temperature,
    )

    for event in response.events():
        if event.event == "add":
            print(event.data)
        elif event.event == "error" or event.event == "interrupted":
            print(event.data)
        elif event.event == "finish":
            print(event.data)
            print(event.meta)
        else:
            print(event.data)


# your api key
zhipuai.api_key = "a1a7a2612a983dadf6add363d149e8ee.DO80Q5q58mrcNHGC"
model_path = {
    "chatglm_pro": "https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_pro/sse-invoke",
    "chatglm_lite": "https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_pro/sse-invoke",

}

#{'code': 200, 'msg': '操作成功', 'data': {'request_id': '8059177926762529198', 'task_id': '516016983114764598059177926762529200', 'task_status': 'PROCESSING'}, 'success': True}

# pro_response={'code': 200, 'msg': '操作成功', 'data': {'request_id': '8059177995482025341', 'task_id': '8059177995482025341', 'task_status': 'SUCCESS', 'choices': [{'role': 'assistant', 'content': '" 人工智能是一种计算机科学领域，它致力于研究和开发计算机系统，使其能够模拟和扩展人的智能能力。人工智能技术通过模拟人类的思维和决策过程，可以实现许多需要人类智能才能完成的任务，如语音识别、图像识别、语言处理、机器学习等。\\n\\n人工智能的发展可以追溯到20世纪50年代和60年代，但直到最近几年，随着计算机算力的不断提高和大数据的普及，人工智能才真正开始进入到人们的生活中。现在，人工智能的技术已经非常成熟，例如深度学习、神经网络、自然语言处理等。\\n\\n人工智能在许多领域具有广泛的应用，如自动驾驶汽车、语音识别、机器人、医疗保健、金融服务、社交媒体、游戏等。通过人工智能技术，我们可以提高生产力、改进医疗保健、提高金融服务效率、改善社交媒体体验等。"'}], 'usage': {'prompt_tokens': 547, 'completion_tokens': 168, 'total_tokens': 715}}, 'success': True}
# std_response={'code': 200, 'msg': '操作成功', 'data': {'request_id': '8059180057066460371', 'task_id': '8059180057066460371', 'task_status': 'SUCCESS', 'choices': [{'role': 'assistant', 'content': '" [\\n  {\\n    \\"head\\": \\"鹰之巢站\\",\\n    \\"relation\\": \\"属于\\",\\n    \\"tail\\": \\"东日本旅客铁道（JR东日本）奥羽本线\\"\\n  },\\n  {\\n    \\"head\\": \\"鹰之巢站\\",\\n    \\"relation\\": \\"位于\\",\\n    \\"tail\\": \\"日本秋田县北秋田市松叶町\\"\\n  },\\n  {\\n    \\"head\\": \\"鹰之巢站\\",\\n    \\"relation\\": \\"类型\\",\\n    \\"tail\\": \\"铁路车站\\"\\n  },\\n  {\\n    \\"head\\": \\"鹰之巢站\\",\\n    \\"relation\\": \\"别名\\",\\n    \\"tail\\": \\"秋田内陆纵贯铁道秋田内陆线鹰巢站\\"\\n  }\\n]"'}], 'usage': {'prompt_tokens': 121, 'completion_tokens': 172, 'total_tokens': 293}}, 'success': True}
# lite_response={'code': 200, 'msg': '操作成功', 'data': {'request_id': '8059178339079437940', 'task_id': '8059178339079437940', 'task_status': 'SUCCESS', 'choices': [{'role': 'assistant', 'content': '" 根据您提供的信息，我可以为您抽取以下关系三元组：\\n\\n1. {  \\n   \'head\': \'鹰之巢站\',  \\n   \'relation\': \'位于\',  \\n   \'tail\': \'日本秋田县北秋田市松叶町\'  \\n}\\n\\n2. {  \\n   \'head\': \'鹰之巢站\',  \\n   \'relation\': \'属于\',  \\n   \'tail\': \'东日本旅客铁道（JR 东日本）奥羽本线\'  \\n}\\n\\n3. {  \\n   \'head\': \'秋田内陆纵贯铁道秋田内陆线\',  \\n   \'relation\': \'相邻\',  \\n   \'tail\': \'鹰巢站\'  \\n}\\n\\n以上是可能的关系三元组，您可以根据需要进行选择和组合。"'}], 'usage': {'prompt_tokens': 121, 'completion_tokens': 163, 'total_tokens': 284}}, 'success': True}


# TODO：
def main(
        load_8bit: bool = False,
        base_model: str = "chatglm_lite",
        max_new_tokens=512,
        temperature=0.9,
        num_beams=3,
        top_p=0.7,
        top_k=40,
        samples=100,
        max_len=1000,
        demo_shot=1,
        seed=42,
        if_icl: bool = False,
        icl_method: str = "random_select",
        interactive: bool = False,
):
    # DataFormat/output/processed_prompt/KnowLM-IE_500_max_1000_seed_42.jsonl
    # DataFormat / output / processed_prompt / KnowLM - IE_500_max_1000_seed_42_shot_1_random_select.jsonl
    if if_icl:
        load_file = "KnowLM-IE_{}_max_{}_seed_{}_shot_{}_{}.jsonl".format(samples, max_len, seed, demo_shot, icl_method)
    else:
        load_file = "KnowLM-IE_{}_max_{}_seed_{}.jsonl".format(samples, max_len, seed)

    model_name = base_model.split('/')[-1]
    data_path = os.path.join("./data/raw",load_file)
    # random.seed(seed)
    # result_collection_path = "./KnowLM-IE-{}-max200-1.json".format(str(cases_samples))
    result_collection_path = "./data/processed/" + os.path.basename(data_path).split('.')[0] + "_" + model_name + "_" + "inference" + ".json"
    print("save infer path in ", result_collection_path)
    cases_golden = []
    cases = []

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as file:
        cases_from_file = [json.loads(line) for line in file]

    for item in cases_from_file:
        cases.append(item["inputs"])
        cases_golden.append(item["golden_label"])
    output_cases = []

    print(f"{'=' * 30}START{'=' * 30}")

    start_time = datetime.now()
    for (inputs, golden_label) in tqdm(zip(cases, cases_golden)):
        # " 好的，根据您提供的信息，我可以为您抽取以下关系三元组：\n\n* 鹰之巢站属于东日本旅客铁道（JR 东日本）  \n* 鹰之巢站位于日本秋田县北秋田市松叶町  \n* 鹰之巢站是东日本旅客铁道（JR 东日本）奥羽本线的铁路车站"
        # " 好的，根据您提供的信息，我可以为您抽取以下关系三元组：\n\n* 鹰之巢站属于东日本旅客铁道（JR 东日本）  \n* 鹰之巢站位于日本秋田县北秋田市松叶町  \n* 鹰之巢站是东日本旅客铁道（JR 东日本）奥羽本线的铁路车站"
        # " [\n  {\n    \"head\": \"鹰之巢站\",\n    \"relation\": \"属于\",\n    \"tail\": \"东日本旅客铁道（JR东日本）奥羽本线\"\n  },\n  {\n    \"head\": \"鹰之巢站\",\n    \"relation\": \"位于\",\n    \"tail\": \"日本秋田县北秋田市松叶町\"\n  },\n  {\n    \"head\": \"鹰之巢站\",\n    \"relation\": \"类型\",\n    \"tail\": \"铁路车站\"\n  },\n  {\n    \"head\": \"鹰之巢站\",\n    \"relation\": \"别名\",\n    \"tail\": \"秋田内陆纵贯铁道秋田内陆线的鹰巢站\"\n  }\n]"

        infer_out,tokens_cal = invoke_example(inputs, base_model, top_p=top_p, temperature=temperature)
        # print("#"*100)
        output_cases.append(
            {"inputs": inputs, "outputs": infer_out, "golden_label": golden_label, "tokens_cal": tokens_cal})
        # raise NotImplementedError

    print("total {} samples! Inference cost {} ".format(len(cases), datetime.now() - start_time))

    with open(result_collection_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(json.dumps(ins, ensure_ascii=False) for ins in output_cases))
    print("write in ", result_collection_path)


if __name__ == "__main__":

    fire.Fire(main)
