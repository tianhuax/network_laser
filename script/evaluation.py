import sys
import os
# 定义模型目录路径（必须是模型加载器使用的路径）
# 注意：如果您的 evaluation_utils.py 文件不直接访问 model_dir 变量，
# 您可能需要将 model_dir 的值硬编码到这里，或者通过参数传递。
# 这里使用您主脚本中使用的路径：
# ...
MOE_MODEL_DIR = "/scratch/tx856/moe/balance/model/deepseek"

# 临时将该目录添加到 Python 模块搜索路径
if MOE_MODEL_DIR not in sys.path:
    sys.path.append(MOE_MODEL_DIR)

try:
    # 这一步将导入并执行 modeling_deepseek.py 文件
    # 在执行过程中，modeling_deepseek.py 内部的 'from configuration_deepseek import...' 
    # 现在能通过 sys.path 找到 configuration_deepseek.py。
    from modeling_deepseek import get_expert_load_dict, clear_expert_load_dict
    print(f"成功从本地模型目录 {MOE_MODEL_DIR} 动态导入专家负载函数。")

except ImportError as e:
    print(f"致命错误：无法导入 modeling_deepseek 中的函数。请确认：")
    print(f"1. 路径 {MOE_MODEL_DIR} 是否正确。")
    print(f"2. modeling_deepseek.py 文件中是否定义了 get_expert_load_dict 和 clear_expert_load_dict。")
    print(f"错误信息: {e}")
    # 安全起见，如果导入失败，移除路径
    if MOE_MODEL_DIR in sys.path:
        sys.path.remove(MOE_MODEL_DIR)
    raise # 抛出错误，因为没有这些函数评估无法进行

# 3. （可选）执行完导入后，移除路径。
if MOE_MODEL_DIR in sys.path:
    sys.path.remove(MOE_MODEL_DIR)
# -----------------------------------------------------------

GLOBAL_LOAD_TENSOR = None

import argparse
import os
import numpy
import torch
from torch import nn
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils.evaluation_utils import evaluation

model_dir = "/scratch/tx856/moe/balance/model/deepseek"

config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

# 手动设置 LASER 参数到配置中
# DeepSeek-MoE-16b-chat k=6 (Arc-Challenge/Easy 示例)
config.laser_enabled = False
config.laser_c = 12
config.laser_epsilon_high = 0.10
config.laser_t_fix = 0.30

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    config=config,
    torch_dtype="auto",
    device_map="auto",  # 多 GPU 会自动分配
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print("Tokenizer loaded.")

print(model.device)

model.eval()

# 1. 清除历史负载数据 (确保每次评估都是新的)
clear_expert_load_dict()

# quantize model
# rtn(model, quantization_bit=4, rcf=False, no_trick=True)
ppl_results, task_results, clean_results = evaluation(model=model,
                                                        tokenizer=tokenizer,
                                                        tasks="arc_easy,arc_challenge",    #"piqa,hellaswag,mmlu,winogrande",
                                                        ppls="Self-GRIT/wikitext-2-raw-v1-preprocessed",    #ppls="allenai/c4,Self-GRIT/wikitext-2-raw-v1-preprocessed",
                                                        quantization_bit=4,
                                                        limit=0.2,
                                                        is_baseline=True)
print(ppl_results)
print(clean_results)