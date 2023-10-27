"""
TODO: 1. token计算方式

"""
prefix = '''命名实体识别：抽取文本中的 人名，地点，组织 这三类命名实体，并按照json格式返回结果。

下面是一些范例：

小明对小红说:"你听说过安利吗？" -> {"人名": ["小明","小红"], "组织": ["安利"]}
现在，每年有几十万中国人到美国访问，几千名中国留学生到美国就学。 -> {"地点": ["中国", "美国"]}
中国是联合国安理会常任理事国之一。 -> {"地点": ["中国"], "组织": ["联合国"]}

请对下述文本进行实体抽取，返回json格式。

'''

def get_prompt(text):
    return prefix+text+' -> '

def get_message(prompt,response):
    return [{"role": "user", "content": f'{prompt} -> '},
            {"role": "assistant", "content": response}]


messages  = [{"role": "user", "content": get_prompt("一些摩洛哥球迷已按捺不住，在看台上欢呼雀跃")}]
response = model.chat(tokenizer, messages)
print(response)


{"地点":["摩洛哥"], "组织":[]}

messages = messages+[{"role": "assistant", "content": "{'地点': ['摩洛哥']}"}]
messages.extend(get_message("这次轮到北京国安队，不知会不会再步后尘？","{'组织': ['北京国安队']}"))
messages.extend(get_message("革命党人孙中山在澳门成立同盟会分会","{'人名': ['孙中山'], '地名': ['澳门'], '组织': ['同盟会']}"))
messages.extend(get_message("我曾在安徽芜湖市和上海浦东打工。","{'地点': ['安徽芜湖市', '上海浦东']}"))
display(messages)




def predict(text,temperature=0.01):
    model.generation_config.temperature=temperature
    response = model.chat(tokenizer, 
                          messages = messages+[{'role':'user','content':f'{text} -> '}])
    return response


predict('杜甫是李白的粉丝。') 

"{'人名': ['杜甫', '李白']}"