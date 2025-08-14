# Qwen训练系统 - 简化版使用指南

## 概述

这是Qwen大型语言模型训练系统的简化版本，专注于核心功能，去除了复杂的GPU管理逻辑，让您可以直接指定要使用的GPU进行训练。

## 核心文件

- `train_qwen_simple.py` - 简化版训练脚本
- `run_training_simple.sh` - 简化版启动脚本
- `training_config_simple.json` - 简化版配置文件

## 快速开始

### 1. 查看可用GPU

```bash
# 查看GPU信息
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### 2. 使用启动脚本（推荐）

```bash
# 基本语法
./run_training_simple.sh [stage] [gpu_ids] [experiment_name]

# 示例：使用GPU 0,1,2,3 进行阶段1训练
./run_training_simple.sh 1 "0,1,2,3" my_experiment

# 示例：使用GPU 0 进行单GPU调试
./run_training_simple.sh 1 "0" debug_test

# 示例：使用GPU 4,5,6,7 进行阶段2训练
./run_training_simple.sh 2 "4,5,6,7" stage2_experiment
```

### 3. 直接使用Python脚本

```bash
# 指定GPU ID
python train_qwen_simple.py --gpu_ids 0 1 2 3 --stage 1 --experiment_name my_experiment

# 单GPU训练
python train_qwen_simple.py --gpu_ids 0 --stage 1 --experiment_name single_gpu_test

# 使用所有GPU
python train_qwen_simple.py --stage 1 --experiment_name all_gpu_test
```

## 参数说明

### 训练阶段

- **阶段1** (`stage=1`): 冻结预训练层，只训练RoPE修改的层（推荐先运行）
- **阶段2** (`stage=2`): 解冻全模型，进行整体微调

### GPU配置

- **gpu_ids**: 指定要使用的GPU ID列表
  - 单GPU: `"0"`
  - 多GPU: `"0,1,2,3"`
  - 如果不指定，将使用所有可用GPU

### 实验名称

- **experiment_name**: 实验名称，用于创建输出目录
  - 如果不指定，将自动生成时间戳名称

## 配置文件使用

### 1. 修改配置文件

编辑 `training_config_simple.json`：

```json
{
  "training_args": {
    "gpu_ids": [0, 1, 2, 3],  // 指定要使用的GPU
    "stage": 1,               // 训练阶段
    "experiment_name": "my_experiment"
  }
}
```

### 2. 使用配置文件启动

```bash
python train_qwen_simple.py training_config_simple.json
```

## 常用配置示例

### 单GPU调试
```bash
./run_training_simple.sh 1 "0" debug_test
```

### 4个A800 GPU训练（推荐大模型）
```bash
./run_training_simple.sh 1 "0,1,2,3" a800_experiment
```

### 4个A40 GPU训练（标准训练）
```bash
./run_training_simple.sh 1 "4,5,6,7" a40_experiment
```

### 使用所有8个GPU
```bash
./run_training_simple.sh 1 "0,1,2,3,4,5,6,7" full_gpu_experiment
```

## 输出目录结构

训练完成后，输出目录结构如下：

```
/work/xiaoyonggao/[experiment_name]/
├── pytorch_model.bin          # 训练好的模型
├── config.json               # 模型配置
├── tokenizer.json            # 分词器
├── training_args.bin         # 训练参数
├── trainer_state.json        # 训练状态
├── logs/                     # TensorBoard日志
├── train_command.txt         # 训练命令记录
└── training_info.txt         # 训练信息记录
```

## 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir=/work/xiaoyonggao/[experiment_name]/logs

# 在浏览器中访问
http://localhost:6006
```

### 查看训练日志

```bash
# 实时查看训练日志
tail -f /work/xiaoyonggao/[experiment_name]/training.log
```

## 故障排除

### 1. GPU内存不足

- 减少 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`

### 2. GPU ID无效

- 检查GPU ID是否在有效范围内（0-7）
- 确保指定的GPU没有被其他进程占用

### 3. 训练中断

- 训练会自动保存检查点
- 重新运行相同命令即可从断点继续

## 与原版本的区别

| 功能 | 原版本 | 简化版 |
|------|--------|--------|
| GPU选择 | 复杂的类型选择和自动检测 | 直接指定GPU ID |
| 配置方式 | 多种配置选项 | 简化的配置选项 |
| 启动方式 | 多个参数 | 3个核心参数 |
| 学习曲线 | 较陡峭 | 平缓 |
| 功能完整性 | 完整 | 核心功能 |

## 总结

简化版训练系统专注于核心功能：

1. **简单直接**: 只需指定GPU ID即可开始训练
2. **易于理解**: 去除复杂的GPU管理逻辑
3. **快速上手**: 3个参数即可启动训练
4. **功能完整**: 保留所有核心训练功能

适合快速原型开发和简单的训练任务。如需更高级的功能，请使用完整版本。