import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import torch.nn as nn


## 模型加载
#使用QLoRA引入的 NF4量化数据类型以节约显存
model_name_or_path ='/work/ytw/LLM/Baichuan2-13B-Chat' #远程 'baichuan-inc/Baichuan-13B-Chat'

bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

tokenizer = AutoTokenizer.from_pretrained(
   model_name_or_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True) 

model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

# 测试用例
from IPython.display import clear_output 
messages = []
messages.append({"role": "user",
                 "content": "世界上第二高的山峰是哪座?"})
response = model.chat(tokenizer,messages=messages,stream=True)
for res in response:
    print(res)
    clear_output(wait=True)

prefix = '''命名实体识别：抽取文本中的 人名，地点，组织 这三类命名实体，并按照json格式返回结果。

下面是一些范例：

小明对小红说:"你听说过安利吗？" -> {"人名": ["小明","小红"], "组织": ["安利"]}
现在，每年有几十万中国人到美国访问，几千名中国留学生到美国就学。 -> {"地点": ["中国", "美国"]}
中国是联合国安理会常任理事国之一。 -> {"地点": ["中国"], "组织": ["联合国"]}

请对下述文本进行实体抽取，返回json格式。

'''

## 构造简单的few-shot prompt
def get_prompt(text):
    return prefix+text+' -> '

def get_message(prompt,response):
    return [{"role": "user", "content": f'{prompt} -> '},
            {"role": "assistant", "content": response}]


messages  = [{"role": "user", "content": get_prompt("一些摩洛哥球迷已按捺不住，在看台上欢呼雀跃")}]
response = model.chat(tokenizer, messages)
print(response)


# {"地点":["摩洛哥"], "组织":[]}

messages = messages+[{"role": "assistant", "content": "{'地点': ['摩洛哥']}"}]
messages.extend(get_message("这次轮到北京国安队，不知会不会再步后尘？","{'组织': ['北京国安队']}"))
messages.extend(get_message("革命党人孙中山在澳门成立同盟会分会","{'人名': ['孙中山'], '地名': ['澳门'], '组织': ['同盟会']}"))
messages.extend(get_message("我曾在安徽芜湖市和上海浦东打工。","{'地点': ['安徽芜湖市', '上海浦东']}"))
print(messages)


def predict(text,temperature=0.01):
    model.generation_config.temperature=temperature
    response = model.chat(tokenizer, 
                          messages = messages+[{'role':'user','content':f'{text} -> '}])
    return response

print(predict('杜甫是李白的粉丝。') )

from sklearn.model_selection import train_test_split
import pandas as pd 
from tqdm import tqdm

df = pd.read_pickle('dfner_13k.pkl')
dfdata,dftest = train_test_split(df,test_size=300,random_state=42)
dftrain,dfval = train_test_split(dfdata,test_size=200,random_state=42)

preds = ['' for x in dftest['target']]
for i in tqdm(range(len(preds))):
    preds[i] = predict(dftest['text'].iloc[i])
    

def toset(s):
    try:
        dic = eval(str(s))
        res = []
        for k,v in dic.items():
            for x in v:
                if x:
                    res.append((k,x))
        return set(res)
    except Exception as err:
        print(err)
        return set()

dftest['pred'] = [toset(x) for x in preds]
dftest['gt'] = [toset(x) for x in dftest['target']]
dftest['tp_cnt'] = [len(pred&gt) for pred,gt in zip(dftest['pred'],dftest['gt'])]
dftest['pred_cnt'] = [len(x) for x in dftest['pred']]
dftest['gt_cnt'] = [len(x) for x in dftest['gt']]

precision = sum(dftest['tp_cnt'])/sum(dftest['pred_cnt'])
print('precision = '+str(precision))

recall = sum(dftest['tp_cnt'])/sum(dftest['gt_cnt'])
print('recall = '+str(recall))

f1 = 2*precision*recall/(precision+recall)
print('f1_score = '+str(f1))