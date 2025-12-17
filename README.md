# LASER: Load-Balanced MoE Routing Reproduction

This repository provides code to reproduce part of the experiments from the paper:

**From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing**  
📄 https://arxiv.org/pdf/2510.03293

The code focuses on evaluating **load balance behavior and task accuracy** for Mixture-of-Experts (MoE) models under both **LASER routing** and **baseline routing**.

---

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.9+
- PyTorch with CUDA support
- HuggingFace `transformers`
- Other dependencies as required by the project

---

### 1.2 Model Preparation (Required)

You must download the **DeepSeek MoE base model** locally:
'''
deepseek-ai/deepseek-moe-16b-base
'''
Download it using HuggingFace tools (e.g., `huggingface-cli`) or manually, and place it in a **local directory** of your choice.

---

### 1.3 Replace Model Implementation

After downloading the model, replace the default DeepSeek model implementation:

- Locate the downloaded model directory
- Replace the file:
`
modeling_deepseek.py
`
with the provided version in this repository:
`
$PROJ_ROOT$/model/deepseek/modeling_deepseek.py
`

This modification is required to enable LASER routing and balance control.

---

## 2. Running the Experiments

### 2.1 Run Evaluation

From the project root directory, run:

```bash
python -m script.evaluation
```

The routing mode is controlled by a single flag in `modeling_deepseek.py` (line 349).

---

### 2.2 Output
At the end of execution, the program prints:
- Final imbalance factor
- Task accuracy



