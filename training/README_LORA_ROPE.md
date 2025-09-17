# LoRA + RoPE 训练脚本使用说明

## 概述

`train_lora_rope.py` 是基于原始 `train_simple.py` 开发的增强版训练脚本，主要特性：

1. **LoRA微调**: 使用PEFT库实现高效的LoRA (Low-Rank Adaptation) 微调
2. **RoPE层去除**: 支持指定层的RoPE (Rotary Position Embedding) 禁用
3. **内存优化**: LoRA大幅减少显存占用和训练时间
4. **兼容性**: 与原始配置系统完全兼容

## 主要改进

### 1. LoRA集成
- 使用PEFT库实现LoRA微调
- 默认配置：rank=16, alpha=32, dropout=0.1
- 目标模块：q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 大幅减少可训练参数数量

### 2. RoPE层控制
- 保持原有的RoPE层禁用功能
- 在LoRA应用之前进行RoPE patch
- 支持通过配置文件指定禁用的层

### 3. 训练优化
- 只保存LoRA权重，节省存储空间
- 支持梯度检查点，进一步优化内存
- 兼容原有的数据集加载和训练配置

## 使用方法

### 1. 基本使用

```bash
cd /home/xiaoyonggao/Flare/training
python train_lora_rope.py
```

### 2. 配置文件

使用与原始脚本相同的 `simple_config.json` 配置文件：

```json
{
  "model": {
    "model_name": "Qwen/Qwen2.5-3B",
    "no_rope_layers": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    "rope_theta": 1000000.0,
    "use_flash_attention": true,
    "torch_dtype": "bfloat16"
  },
  "data": {
    "dataset_name": "togethercomputer/RedPajama-Data-1T-Sample",
    "dataset_size": 10000,
    "validation_split": 0.1,
    "max_length": 8096
  },
  "training": {
    "learning_rate": 1e-4,
    "batch_size": 2,
    "num_epochs": 3,
    "warmup_steps": 100
  }
}
```

### 3. 测试脚本

运行测试脚本验证功能：

```bash
python test_lora_rope.py
```

## 输出结果

### 1. 模型保存
- LoRA权重保存在 `/work/xiaoyonggao/{experiment_name}_lora/final_lora_model/`
- 包含adapter_config.json和adapter_model.safetensors
- 训练配置保存在training_config.json

### 2. 使用LoRA模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "/path/to/lora/model")

# 推理
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs)
```

## 技术细节

### 1. LoRA配置
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,          # LoRA scaling
    lora_dropout=0.1,       # Dropout率
    target_modules=[        # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none"
)
```

### 2. RoPE Patch时机
1. 加载基础模型
2. 应用RoPE patch (禁用指定层)
3. 准备模型进行LoRA训练
4. 应用LoRA配置

### 3. 内存优化
- 使用`prepare_model_for_kbit_training`优化内存
- 梯度检查点减少激活值存储
- 只训练LoRA参数，大幅减少显存需求

## 性能对比

| 方法 | 可训练参数 | 显存占用 | 训练速度 | 模型大小 |
|------|------------|----------|----------|----------|
| 全量微调 | 100% | 高 | 慢 | 大 |
| LoRA微调 | ~1% | 低 | 快 | 小 |

## 注意事项

1. **依赖要求**: 确保安装了PEFT库 (`pip install peft`)
2. **模型兼容**: 主要支持Qwen2.5系列模型
3. **RoPE层**: 确保指定的no_rope_layers在模型层数范围内
4. **显存**: LoRA虽然减少显存，但仍需要足够显存加载基础模型

## 故障排除

### 1. PEFT导入错误
```bash
pip install peft==0.17.0
```

### 2. CUDA内存不足
- 减少batch_size
- 启用梯度检查点
- 使用更小的LoRA rank

### 3. RoPE层配置错误
- 检查no_rope_layers是否在有效范围内
- 确保模型支持RoPE patch

## 扩展功能

### 1. 自定义LoRA配置
修改`get_lora_config()`函数来调整LoRA参数：

```python
def get_lora_config():
    return LoraConfig(
        r=32,              # 增加rank提高表达能力
        lora_alpha=64,     # 相应调整alpha
        target_modules=[   # 自定义目标模块
            "q_proj", "v_proj"  # 只训练部分模块
        ]
    )
```

### 2. 多GPU训练
脚本支持多GPU训练，通过配置文件的gpu_ids设置：

```json
{
  "environment": {
    "gpu_ids": [0, 1, 2, 3]
  }
}
```

## 总结

LoRA + RoPE训练脚本提供了一个高效、灵活的微调解决方案，特别适合：
- 资源受限的环境
- 需要快速实验的场景
- 大模型的特定任务适配
- RoPE机制的研究和实验