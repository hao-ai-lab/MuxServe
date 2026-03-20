# 新建test_param_name.py，运行查看参数名
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型（CPU即可，仅看参数名）
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16)

# 打印embedding层参数名
for name, param in model.named_parameters():
    if "embed" in name or "token" in name:
        print("Embedding参数名:", name)