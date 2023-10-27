
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


## 构造简单的few-shot prompt
def get_prompt(text):
    return prefix+text+' -> '

def get_message(prompt,response):
    return [{"role": "user", "content": f'{prompt} -> '},
            {"role": "assistant", "content": response}]

prefix = '''命名实体识别：抽取文本中的 人名，地点，组织 这三类命名实体，并按照json格式返回结果。

下面是一些范例：

小明对小红说:"你听说过安利吗？" -> {"人名": ["小明","小红"], "组织": ["安利"]}
现在，每年有几十万中国人到美国访问，几千名中国留学生到美国就学。 -> {"地点": ["中国", "美国"]}
中国是联合国安理会常任理事国之一。 -> {"地点": ["中国"], "组织": ["联合国"]}

请对下述文本进行实体抽取，返回json格式。

'''



model_name = '/work/ytw/LLM/Baichuan2-7B-Base'
adapter_name = '/work/ytw/Firefly-master/output/firefly-baichuan2-7b-1.1M-epoch-1-50k-1010/final'

# sampling_params = SamplingParams(temperature=0.2, top_k=20, top_p=0.85, presence_penalty=0.3, frequency_penalty=0.5, max_tokens=1024)
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.1 # 
device = 'cuda'
input_pattern = '<s>{}</s>'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, adapter_name)
model.eval()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)


text = input("""User：命名实体识别：抽取文本中的 人名，地点，组织 这三类命名实体，并按照json格式返回结果。

下面是一些范例：

小明对小红说:"你听说过安利吗？" -> {"人名": ["小明","小红"], "组织": ["安利"]}
现在，每年有几十万中国人到美国访问，几千名中国留学生到美国就学。 -> {"地点": ["中国", "美国"]}
中国是联合国安理会常任理事国之一。 -> {"地点": ["中国"], "组织": ["联合国"]}

请对下述文本进行实体抽取，返回json格式。
""")
while True:
    text = input_pattern.format(text)
    # text = input_pattern.format(prefix+'\n'+text)
    print("input:",text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
        top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )
    rets = tokenizer.batch_decode(outputs)
    output = rets[0].strip().replace(text, "").replace('</s>', "")
    print("Firefly：{}".format(output))
    text = input('User：')