# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     inference_zhiyan
   Author :       MSI-NB
   date：          2023/10/26
   Change Activity:
                   2023/10/26:
-------------------------------------------------
   功能：fastapi调用形式调用 zhiyan-13B
-------------------------------------------------
"""

from transformers import pipeline, set_seed
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import os
import re, json,fire

from tqdm import tqdm
import random
import requests
import collections
from datetime import datetime


def getZhiyan13_single(query):
    # 第一版：
    url = 'http://192.168.200.211:8009/textgen?'
    # url = 'http://192.168.200.160:25684/textgen?'

    params = {'input': query}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    r = requests.get(url, params=params, headers=headers)
    full_list = r.json()
    return full_list


# TODO：1.rp需要更改写法或者更改命名获取方式
# 2. 测试zhiyan推理效果和速度
# 3. 推理后的结果解析
def main(
        load_8bit: bool = False,
        base_model: str = "zhiyan-13B",
        max_new_tokens=512,
        temperature=0.3,
        num_beams=3,
        top_p=0.75,
        top_k=40,
        repetition_penalty=1.1,
        samples=100,
        max_len=1000,
        demo_shot=1,
        seed=42,
        if_icl: bool = False,
        icl_method: str = "random_select",
        interactive: bool = False,
):
    # @@@ 自定义cases 推理测试，并保存结果
    # cases_samples = 500
    # DataFormat/output/processed_prompt/KnowLM-IE_500_max_1000_seed_42.jsonl
    if if_icl:
        load_file = "KnowLM-IE_{}_max_{}_seed_{}_shot_{}_{}.jsonl".format(samples, max_len, seed, demo_shot, icl_method)
    else:
        load_file = "KnowLM-IE_{}_max_{}_seed_{}.jsonl".format(samples, max_len, seed)

    model_name = base_model.split('/')[-1]
    data_path = "./" + load_file
    # random.seed(seed)
    lll="E:\MY_3090_CODE\MUIE\DataFormat\output\processed_prompt\KnowLM-IE_100_max_1000_seed_42.jsonl"
    result_collection_path = "./" + os.path.basename(data_path).split('.')[0] + "_" + model_name + "_" + "rp_" + str(
        repetition_penalty*10) + "_" + "inference" + ".json"
    print("save infer path in ", result_collection_path)
    # result_collection_path = "./KnowLM-IE-{}-max200-1.json".format(str(cases_samples))
    cases_golden = []
    cases = []

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as file:
        cases_from_file = [json.loads(line) for line in file]

    for item in cases_from_file:
        cases.append(item["inputs"])
        cases_golden.append(item["golden_label"])
    output_cases = []

    print(f"{'=' * 30}INFO{'=' * 31}")
    print("zhixi-13b loaded successfully, the next is case :)")
    print(f"{'=' * 30}START{'=' * 30}")

    start_time = datetime.now()
    for (inputs, golden_label) in tqdm(zip(cases, cases_golden)):
        # print(f"Output: {evaluate(input=inputs)}")
        tokens_cal=-1
        # print("inputs: ",inputs)
        infer_out = getZhiyan13_single(inputs)
        infer_out= infer_out['output']
        # print("output: ",infer_out)
        # print("#"*100)
        output_cases.append(
            {"inputs": inputs, "outputs": infer_out, "golden_label": golden_label, "tokens_cal": tokens_cal})
    print("total {} samples! cost {} ".format(len(cases), datetime.now() - start_time))

    with open(result_collection_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(json.dumps(ins, ensure_ascii=False) for ins in output_cases))
    print("write in ", result_collection_path)

if __name__ == "__main__":
    fire.Fire(main)


