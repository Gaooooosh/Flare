# Qwen模型训练迁移指南

本文档介绍了新的多卡训练系统的使用方法，该系统基于原有的 `train_qwen-stage.py` 脚本进行了全面升级。

## 🚀 新功能特性

### ✅ 已实现功能
- **多GPU训练支持**: 支持DDP/FSDP多卡并行训练
- **灵活GPU配置**: 支持A800、A40等不同GPU类型的自动选择
- **Hugging Face集成**: 完全集成HF生态系统，支持在线数据集下载
- **TensorBoard记录**: 完整的训练过程可视化
- **统一输出管理**: 所有文件统一保存到 `/work/xiaoyonggao` 目录
- **完善评估系统**: 支持PPL等多种评估指标
- **模型修改兼容**: 完全兼容原有的RoPE禁用功能
- **内存优化**: 自动内存管理和优化
- **早停机制**: 防止过拟合的智能早停
- **阶段化训练**: 支持两阶段训练策略

## 📁 文件结构

```
/home/xiaoyonggao/Flare/
├── train_qwen_multi_gpu.py      # 新的多卡训练脚本
├── evaluate_model_enhanced.py   # 增强版评估脚本
├── training_config.json         # 训练配置文件
├── run_training.sh             # 一键启动脚本
├── patch_qwen_rope.py          # 模型修改脚本（原有）
├── train_qwen-stage.py         # 原始训练脚本（保留）
└── README_MIGRATION.md         # 本文档
```

## 🛠 环境准备

### 1. Python依赖
确保安装了以下依赖包：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install flash-attn --no-build-isolation
pip install tensorboard
pip install numpy tqdm
```

### 2. 目录权限
确保有 `/work/xiaoyonggao` 目录的写入权限：
```bash
mkdir -p /work/xiaoyonggao
chmod 755 /work/xiaoyonggao
```

## 🚀 快速开始

### 方法1: 使用一键启动脚本（推荐）

```bash
# 阶段1训练（冻结预训练层）
./run_training.sh 1 A800 my_experiment_stage1

# 阶段2训练（全模型微调）
./run_training.sh 2 A800 my_experiment_stage2
```

参数说明：
- 第1个参数：训练阶段（1或2）
- 第2个参数：GPU类型（A800、A40或auto）
- 第3个参数：实验名称（可选）

### 方法2: 直接使用Python脚本

```bash
# 阶段1训练
python train_qwen_multi_gpu.py \
    --stage 1 \
    --gpu_type A800 \
    --experiment_name my_experiment \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 3 \
    --max_steps 1000

# 阶段2训练
python train_qwen_multi_gpu.py \
    --stage 2 \
    --gpu_type A800 \
    --experiment_name my_experiment \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 2 \
    --max_steps 2000 \
    --gradient_checkpointing
```

## 📊 训练监控

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir=/work/xiaoyonggao/your_experiment/tensorboard

# 在浏览器中访问
http://localhost:6006
```

### 实时日志查看
```bash
# 查看训练日志
tail -f /work/xiaoyonggao/your_experiment/training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 🔧 高级配置

### 1. GPU配置

#### 🎯 手动选择GPU

```bash
# 方法1: 使用启动脚本指定GPU
./run_training.sh 1 manual my_experiment "0,1,2,3"

# 方法2: 直接在Python脚本中指定
python train_qwen_multi_gpu.py --gpu_ids 0 1 2 3

# 方法3: 单GPU训练
python train_qwen_multi_gpu.py --gpu_ids 0

# 方法4: 使用混合GPU（A800+A40）
python train_qwen_multi_gpu.py --gpu_ids 0 1 4 5
```

#### 🔍 按类型选择GPU

```bash
# 仅使用A800 GPU
python train_qwen_multi_gpu.py --gpu_type A800
./run_training.sh 1 A800 my_experiment

# 仅使用A40 GPU
python train_qwen_multi_gpu.py --gpu_type A40
./run_training.sh 1 A40 my_experiment

# 自动选择所有可用GPU
python train_qwen_multi_gpu.py --gpu_type auto
./run_training.sh 1 auto my_experiment
```

#### 📊 当前服务器GPU布局

根据系统检测，当前服务器GPU配置：
- **GPU 0-3**: NVIDIA A800 80GB PCIe (高内存)
- **GPU 4-7**: NVIDIA A40 (标准内存)

**推荐使用策略**：
- 大模型/长序列：优先使用A800 (0-3)
- 多实验并行：分组使用，避免资源冲突
- 调试测试：使用单个GPU

#### ⚙️ 配置文件方式

参考 `training_config_gpu_examples.json` 查看详细配置示例：

```json
{
  "training_args": {
    "gpu_ids": [0, 1, 2, 3],  // 手动指定GPU
    "gpu_type": null          // 或指定类型如"A800"
  }
}
```

### 2. 数据集配置

```bash
# 使用Hugging Face数据集
python train_qwen_multi_gpu.py \
    --dataset_name "togethercomputer/RedPajama-Data-1T-Sample" \
    --text_column "text" \
    --max_seq_length 4096

# 限制数据集大小
python train_qwen_multi_gpu.py \
    --dataset_size 100000 \
    --validation_split_percentage 0.1
```

### 3. 模型配置

```bash
# 自定义RoPE设置
python train_qwen_multi_gpu.py \
    --rope_theta 1000000.0 \
    --no_rope_layers 20 21 22 23 24 25 26 27 28 29 30 31 32

# 使用不同的模型
python train_qwen_multi_gpu.py \
    --model_name_or_path "Qwen/Qwen2.5-7B"
```

## 📈 模型评估

### 基础评估
```bash
python evaluate_model_enhanced.py \
    --model_path /work/xiaoyonggao/your_experiment/final_model \
    --use_default_datasets
```

### 自定义评估
```bash
python evaluate_model_enhanced.py \
    --model_path /work/xiaoyonggao/your_experiment/final_model \
    --output_dir /work/xiaoyonggao/evaluation_results \
    --max_length 2048 \
    --batch_size 1 \
    --max_samples_per_dataset 1000
```

### 评估结果
评估完成后，结果将保存在：
- `evaluation_results.json`: 详细的数值结果
- `evaluation_report.md`: 可读性强的报告
- `tensorboard/`: TensorBoard日志

## 📂 输出目录结构

训练完成后，输出目录结构如下：
```
/work/xiaoyonggao/your_experiment/
├── checkpoints/              # 训练检查点
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── final_model/              # 最终模型
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── logs/                     # 各种日志文件
├── tensorboard/              # TensorBoard日志
├── training.log              # 训练日志
├── training_metrics.json     # 训练指标
├── train_command.txt         # 训练命令记录
└── environment_info.txt      # 环境信息
```

## 🔍 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 减小批次大小
   --per_device_train_batch_size 1
   
   # 增加梯度累积步数
   --gradient_accumulation_steps 32
   
   # 启用梯度检查点
   --gradient_checkpointing
   ```

2. **数据集加载失败**
   ```bash
   # 检查网络连接
   ping huggingface.co
   
   # 使用本地缓存
   --cache_dir /path/to/cache
   ```

3. **多卡训练问题**
   ```bash
   # 检查GPU可用性
   python -c "import torch; print(torch.cuda.device_count())"
   
   # 设置DDP超时
   --ddp_timeout 1800
   ```

### 日志分析

```bash
# 查看错误信息
grep -i error /work/xiaoyonggao/your_experiment/training.log

# 查看GPU使用情况
grep -i "gpu\|cuda\|memory" /work/xiaoyonggao/your_experiment/training.log

# 查看训练进度
grep -i "step\|epoch\|loss" /work/xiaoyonggao/your_experiment/training.log
```

## 🔄 从原脚本迁移

### 参数对照表

| 原脚本参数 | 新脚本参数 | 说明 |
|-----------|-----------|------|
| `--model_name_or_path` | `--model_name_or_path` | 相同 |
| `--no_rope_layers` | `--no_rope_layers` | 相同 |
| `--rope_theta` | `--rope_theta` | 相同 |
| `--output_dir` | `--base_output_dir` + `--experiment_name` | 新的目录管理方式 |
| `--stage1_max_seq_len` | `--max_seq_length` | 统一序列长度参数 |
| `CUDA_VISIBLE_DEVICES` | `--gpu_ids` 或 `--gpu_type` | 更灵活的GPU选择 |

### 迁移步骤

1. **备份原有脚本和数据**
   ```bash
   cp train_qwen-stage.py train_qwen-stage.py.backup
   ```

2. **测试新脚本**
   ```bash
   # 小规模测试
   python train_qwen_multi_gpu.py \
       --stage 1 \
       --max_steps 10 \
       --dataset_size 1000
   ```

3. **正式迁移**
   ```bash
   # 使用相同的参数配置
   ./run_training.sh 1 A800 migration_test
   ```

## 📞 技术支持

如果遇到问题，请：

1. 检查日志文件中的错误信息
2. 确认环境配置是否正确
3. 验证GPU和内存资源是否充足
4. 查看本文档的故障排除部分

## 🎯 最佳实践

1. **资源规划**
   - 根据可用GPU内存调整批次大小
   - 使用梯度累积来模拟大批次训练
   - 定期清理检查点以节省存储空间

2. **训练策略**
   - 先进行小规模测试验证配置
   - 使用TensorBoard监控训练过程
   - 设置合理的早停参数防止过拟合

3. **实验管理**
   - 使用有意义的实验名称
   - 记录重要的配置变更
   - 定期备份重要的模型检查点

---

**注意**: 本系统完全兼容原有的模型修改功能，您可以放心迁移现有的训练任务。如有任何问题，请参考故障排除部分或查看详细的日志信息。