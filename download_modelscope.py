from modelscope.hub.snapshot_download import snapshot_download
import os
# 默认位置 ~/.cache/modelscope/hub
# 自定义位置
os.environ['MODELSCOPE_CACHE'] = './'

model_dir = snapshot_download('AI-ModelScope/GAIRMath-Abel-13b', cache_dir='path/to/local/dir') #, revision='v1.0.1'

print(modelscope.version.__release_datetime__)