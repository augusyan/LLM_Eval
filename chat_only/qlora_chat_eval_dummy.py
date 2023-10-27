
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import args
import warnings
import argparse
import logging
import sys

sys.path.append("..")

warnings.filterwarnings("ignore", category=UserWarning)


"""
TODO: 1.自动化输入，构造测试问题集
2. 仿照 input output的方式，将输入输出按照json保存，后续分析
3. 参数化所有需要调参的方法，方便接入shell
4. 一些性能相关的参数记录，例如，每个instance的token数，推理使用的

"""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=8, type=int, help="batch size")  # 32
    parser.add_argument('--lr', default=3e-5, type=float, help="learning rate")
    parser.add_argument('--topk', default=5, type=int, help="cal top k attention score from feature")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=2023, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=4, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    # NER_tw15_multi_fuse_wo_cl_text_loss, NER_tw15_multi_fuse_wo_cl_img_loss, NER_tw15_multi_fuse_wo_cl_all_loss
    # ，NER_tw17_multi_fuse_pure_clip_00，NER_tw17_multi_fuse_add_i2t_cl_01，NER_tw15_multi_fuse_filip_v3_01
    parser.add_argument('--model_name', default='FMNER_source_wiki_MGCL_02', type=str, help="Load model from load_path")
    # parser.add_argument('--model_name', default='NER_multi_fuse_dy_flip2_best_01', type=str, help="Load model from load_path")
    # experiment ablation experiment
    parser.add_argument('--save_path', default='./logs/normal', type=str, help="save model at save_path")
    parser.add_argument('--write_path', default='./logs/normal', type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--only_test', action='store_true', default=False)
    parser.add_argument('--max_seq', default=80, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()
    args.log_path = os.path.join(args.save_path, args.model_name+'_performance.txt')
    
    model_name = '/work/ytw/LLM/Qwen-7B'
    # adapter_name = '/work/ytw/Firefly-master/output/firefly-qwen-7b-uie-epoch-1-50k-1009/final'
    adapter_name = '/work/ytw/Firefly-master/output/firefly-qwen-7b-alpaca-chat-epoch-1-50k-1008/final'

    # sampling_params = SamplingParams(temperature=0.2, top_k=20, top_p=0.85, presence_penalty=0.3, frequency_penalty=0.5, max_tokens=1024)
    max_new_tokens = 500
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0 # 
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

# input:  一些摩洛哥球迷已按捺不住，在看台上欢呼雀跃

while True:
    text = input_pattern.format(text) # 纯输入chat
    # text = input_pattern.format(prefix+'\n'+text)  # 带ner prompt的chat
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
    
    
    torch.cuda.empty_cache()
    


    
if __name__ == "__main__":
    main()

