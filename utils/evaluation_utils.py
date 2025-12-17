import re
import torch
import torch.nn as nn

from datasets import load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Union

from collections import defaultdict
import torch.nn as nn
from typing import Dict, Any

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 使用一个全局张量来模拟实时负载（需要手动更新，或者在最简单的情况下，全部设为 0）
GLOBAL_LOAD_TENSOR = None

# -------------------------------------------------------------
# 1. 全局容器：存储每层 MoE 的负载数据
# 键: Layer名称 (例如: 'model.layers.0.mlp')
# 值: 专家的负载计数 defaultdict(int)
LAYER_WISE_EXPERT_LOADS: Dict[str, defaultdict] = {}
# -------------------------------------------------------------


def create_expert_load_recorder_hook(module_name: str, num_experts: int):
    """
    创建一个前向 Hook 函数。
    它将在 DeepseekMoE 模块执行后被调用，并读取模块存储的路由索引。
    
    Args:
        module_name: 模块的唯一名称 ('model.layers.X.mlp').
        num_experts: 专家总数 (用于验证和边界检查).
    """
    # Hook 函数的签名是固定的：(module, input, output)
    def forward_hook(module: nn.Module, input: Any, output: torch.Tensor):
        
        # 1. 确保在评估模式下运行
        if module.training:
            return output
            
        # 2. 检查 DeepseekMoE 模块是否成功存储了索引
        if not hasattr(module, "_temp_topk_idx"):
            print(f"Warning: Hook found on {module_name}, but _temp_topk_idx not available. Skipping record.")
            return output

        topk_idx = module._temp_topk_idx
        
        # 3. 统计负载
        # topk_idx shape: [seq_len * bsz, num_experts_per_tok]
        flat_topk_idx = topk_idx.view(-1)
        
        # 将统计结果存储到全局容器中
        if module_name not in LAYER_WISE_EXPERT_LOADS:
             LAYER_WISE_EXPERT_LOADS[module_name] = defaultdict(int)

        load_dict = LAYER_WISE_EXPERT_LOADS[module_name]
        
        # 批量统计 (比 Python 循环更快)
        # 确保 flat_topk_idx 在 CPU 上进行列表转换和统计，如果需要在 CPU 上汇总的话
        for idx in flat_topk_idx.cpu().tolist():
            load_dict[idx] += 1
            
        # 4. 清理临时属性 (可选，但推荐)
        delattr(module, "_temp_topk_idx")
        
        return output # Hook 必须返回 output 或 None
        
    return forward_hook


def register_expert_load_hooks(model: nn.Module):
    """
    遍历模型，找到所有的 DeepseekMoE 模块并注册 Hook。
    """
    # 确保清除历史数据
    LAYER_WISE_EXPERT_LOADS.clear()
    
    # 存储 Hook 句柄，以便稍后移除 Hook
    hook_handles = [] 
    
    num_routed_experts = model.config.n_routed_experts
    
    for name, module in model.named_modules():
        # 检查模块类名是否是 DeepseekMoE
        if module.__class__.__name__ == 'DeepseekMoE':
            hook = create_expert_load_recorder_hook(name, num_routed_experts)
            # 注册 Hook。使用 register_forward_hook
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)
            print(f"Registered hook on {name}")

    return hook_handles

@torch.no_grad()
def evaluate_model(model: nn.Module,
                   tokenizer: AutoTokenizer,
                   tasks: str,
                   eval_ppl: str,
                   num_fewshot: int = 0,
                   limit: Union[int, float] = -1,
                   batch_size: int = 1):
    # lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device_map="auto")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    lm.model.eval()
    lm.seqlen = 2048
    ppl_results = {}
    task_results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            if "c4" in dataset:
                testdata = load_dataset(
                    dataset,
                    "en",
                    split="validation",
                )
                testloader = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
                testenc = testloader.input_ids[:, :(256 * lm.seqlen)]
            else:
                testdata = load_dataset(
                    dataset,
                    split="test",
                )
                testloader = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
                testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)].to(lm.device)
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)][:, 1:].to(shift_logits.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            ppl_results[dataset] = ppl.item()
    if tasks != "":
        csr_results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
        )

        csr_results = csr_results["results"]
        task_results.update(csr_results)

    clean_results = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in task_results.items()}

    return ppl_results, task_results, clean_results


def evaluation(model: nn.Module,
               tokenizer: AutoTokenizer,
               tasks: str,
               ppls: str,
               quantization_bit: int,
               limit: float = -1,
               is_baseline: bool = False):

    # 1. 设置 LASER 参数和模式
    # 假设 LASER 参数 c=10, E_high=0.4, t_fix=0.8

    # 遍历所有 DeepseekMoE 模块并设置 LASER 模式
    global GLOBAL_LOAD_TENSOR
    num_experts = model.config.n_routed_experts

    # 初始化负载张量（用于传递给 MoEGate）
    # 这里初始化为零，意味着所有专家初始负载相等，LASER将基于分数和 Tie-breaking 路由。
    GLOBAL_LOAD_TENSOR = torch.zeros(num_experts, dtype=torch.float32).to(model.device) 

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'DeepseekMoE':
            module.gate.laser_enabled = True # 激活 LASER
            module.gate.epsilon_high = 0.40 # 示例值
            module.gate.t_fix = 0.80 # 示例值
            module.gate.candidate_pool_c = 10 # 示例值

            # 注意：在生产环境中，这些参数应按层深度 L 设置
            
    # 1. 注册 Hook，开始收集数据
    hook_handles = register_expert_load_hooks(model)

    try:
        ppl_results, task_results, clean_results = evaluate_model(model,
                                                              tokenizer,
                                                              tasks=tasks,
                                                              eval_ppl=ppls,
                                                              batch_size=128,
                                                              limit=limit)
    finally:
        # 3. 评估完成后，移除所有 Hook，避免干扰其他操作
        for handle in hook_handles:
            handle.remove()

    # 4. 提取负载数据 (直接使用全局容器)
    expert_load_data = LAYER_WISE_EXPERT_LOADS # 提取所有层的数据

    # --- [新增] 设置绘图保存路径 ---
    save_dir = "expert_load_heatmaps_bl"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Visual] 热力图将保存在: {os.path.abspath(save_dir)}")

# 5. 计算并打印每层不平衡因子，并绘制热力图
    if expert_load_data:
        num_routed_experts = model.config.n_routed_experts 
        print("\n--- 专家负载分析 (Layer-wise) ---")
        
        total_routed_tokens_all_layers = 0
        imbalances = [] 

        # [可选] 指定要画图的层，例如 ['layers.0', 'layers.15']。
        # 如果想画所有层，保持列表为空 [] 即可。
        target_plot_layers = [] 

        for layer_name, loads in expert_load_data.items():
            if not loads:
                continue

            # --- A. 计算统计指标 ---
            # 注意：loads 是一个字典 {expert_id: count}，有些冷门专家可能不在字典里，视为0
            # 我们先将其转换为完整的 dense array，方便画图和计算
            dense_loads = np.zeros(num_routed_experts)
            for e_id, count in loads.items():
                if e_id < num_routed_experts:
                    dense_loads[e_id] = count
            
            total_tokens = np.sum(dense_loads)
            if total_tokens == 0: continue

            avg_load = total_tokens / num_routed_experts
            max_load = np.max(dense_loads)
            imbalance_factor_I_L = max_load / (avg_load + 1e-9) # 防止除以0
            
            total_routed_tokens_all_layers += total_tokens
            imbalances.append(imbalance_factor_I_L)
            
            print(f"[{layer_name}] Imbalance I_L: {imbalance_factor_I_L:.2f}x | Total Tokens: {int(total_tokens)}")

            # --- B. 绘制热力图 (Heatmap) ---
            # 只有当列表为空(画所有层) 或者 当前层在目标列表中时才画
            if not target_plot_layers or layer_name in target_plot_layers:
                try:
                    plt.figure(figsize=(10, 8))
                    
                    # 1. 尝试找到一个接近正方形的形状进行 reshape (例如 64 -> 8x8)
                    # 简单的逻辑：找最接近 sqrt 的因子
                    grid_h = int(math.sqrt(num_routed_experts))
                    while num_routed_experts % grid_h != 0:
                        grid_h -= 1
                    grid_w = num_routed_experts // grid_h
                    
                    heatmap_data = dense_loads.reshape(grid_h, grid_w)

                    # 2. 绘制 Seaborn 热力图
                    # annot=True 会在格子里显示数字，如果专家太多(如160个)建议设为False以免看不清
                    # cmap="YlOrRd" 是从黄到红，红色代表负载高
                    ax = sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap="YlOrRd", 
                                     cbar_kws={'label': 'Token Count'})
                    
                    plt.title(f"Expert Load Heatmap - {layer_name}\n(I_L={imbalance_factor_I_L:.2f})")
                    plt.xlabel(f"Experts (Reshaped {grid_h}x{grid_w})")
                    
                    # 3. 保存图片
                    # 将文件名中的点换成下划线，避免文件系统问题
                    safe_name = layer_name.replace('.', '_')
                    save_path = os.path.join(save_dir, f"{safe_name}_heatmap.png")
                    plt.savefig(save_path)
                    plt.close() # 这一步很重要，避免内存泄漏
                    
                except Exception as e:
                    print(f"  [Warning] 画图失败 {layer_name}: {e}")

        # 计算聚合不平衡因子 I_agg
        if imbalances:
            agg_imbalance = sum(imbalances) / len(imbalances)
            print(f"\n**平均聚合不平衡因子 I_agg: {agg_imbalance:.2f}x**")
            print(f"总计路由 Token 次数: {int(total_routed_tokens_all_layers)}")

    return ppl_results, task_results, clean_results