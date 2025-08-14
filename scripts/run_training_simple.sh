#!/bin/bash

# Qwen多卡训练启动脚本 - 简化版
# 使用方法：
# ./run_training_simple.sh [stage] [gpu_ids] [experiment_name]
# 例如：
#   ./run_training_simple.sh 1 "0,1,2,3" my_experiment
#   ./run_training_simple.sh 2 "0,1" debug_test
#   ./run_training_simple.sh 1 "0" single_gpu_test

set -e

# 默认参数
STAGE=${1:-1}
GPU_IDS=${2:-"0"}
EXPERIMENT_NAME=${3:-"qwen_training_$(date +%Y%m%d_%H%M%S)"}
BASE_OUTPUT_DIR="/work/xiaoyonggao"
# CONFIG_FILE将在脚本中动态设置

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Qwen多卡训练启动脚本 (简化版) ===${NC}"
echo -e "${GREEN}训练阶段: ${STAGE}${NC}"
echo -e "${GREEN}使用GPU: ${GPU_IDS}${NC}"
echo -e "${GREEN}实验名称: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}输出目录: ${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}${NC}"
echo ""

# 参数验证
if [[ ! "$STAGE" =~ ^[12]$ ]]; then
    echo -e "${RED}错误: 训练阶段必须是 1 或 2${NC}"
    echo "  阶段1: 冻结预训练层，专攻新模块"
    echo "  阶段2: 解冻全模型，整体微调"
    exit 1
fi

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo -e "${RED}错误: GPU ID格式不正确${NC}"
    echo "  正确格式: \"0\" 或 \"0,1,2,3\""
    exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 找不到Python解释器${NC}"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查必要的文件
if [ ! -f "$PROJECT_ROOT/training/train_qwen_simple.py" ]; then
    echo -e "${RED}错误: 找不到训练脚本 train_qwen_simple.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/utils/patch_qwen_rope.py" ]; then
    echo -e "${RED}错误: 找不到RoPE补丁文件 patch_qwen_rope.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/configs/training_config_simple.json" ]; then
    echo -e "${RED}错误: 找不到配置文件 training_config_simple.json${NC}"
    exit 1
fi

# 显示GPU信息
echo -e "${YELLOW}检查GPU环境...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: 没有检测到CUDA GPU')
"

# 验证指定的GPU是否存在
echo -e "${YELLOW}验证指定的GPU...${NC}"
python -c "
import torch
gpu_ids = [int(x) for x in '$GPU_IDS'.split(',')]
available_gpus = list(range(torch.cuda.device_count()))
for gpu_id in gpu_ids:
    if gpu_id not in available_gpus:
        print(f'错误: GPU {gpu_id} 不存在')
        exit(1)
print(f'✅ 指定的GPU {gpu_ids} 都可用')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU验证失败，请检查GPU ID${NC}"
    exit 1
fi

# 构建训练命令
TRAIN_CMD="python $PROJECT_ROOT/training/train_qwen_simple.py"

# 添加基本参数
TRAIN_CMD="$TRAIN_CMD --config $PROJECT_ROOT/configs/training_config_simple.json"
TRAIN_CMD="$TRAIN_CMD --stage $STAGE"
TRAIN_CMD="$TRAIN_CMD --base_output_dir $BASE_OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"
TRAIN_CMD="$TRAIN_CMD --enable_tensorboard"

# 添加GPU ID参数
GPU_LIST=$(echo "$GPU_IDS" | sed 's/,/ /g')
TRAIN_CMD="$TRAIN_CMD --gpu_ids $GPU_LIST"

# 根据阶段调整参数
if [ "$STAGE" = "1" ]; then
    echo -e "${YELLOW}阶段1: 冻结预训练层，专攻新模块${NC}"
    TRAIN_CMD="$TRAIN_CMD --learning_rate 1e-4"
    TRAIN_CMD="$TRAIN_CMD --per_device_train_batch_size 1"
    TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps 32"
    TRAIN_CMD="$TRAIN_CMD --max_steps 1000"
    TRAIN_CMD="$TRAIN_CMD --eval_steps 50"
    TRAIN_CMD="$TRAIN_CMD --save_steps 100"
    TRAIN_CMD="$TRAIN_CMD --gradient_checkpointing"
elif [ "$STAGE" = "2" ]; then
    echo -e "${YELLOW}阶段2: 解冻全模型，整体微调${NC}"
    TRAIN_CMD="$TRAIN_CMD --learning_rate 3e-5"
    TRAIN_CMD="$TRAIN_CMD --per_device_train_batch_size 1"
    TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps 32"
    TRAIN_CMD="$TRAIN_CMD --max_steps 2000"
    TRAIN_CMD="$TRAIN_CMD --eval_steps 20"
    TRAIN_CMD="$TRAIN_CMD --save_steps 50"
    TRAIN_CMD="$TRAIN_CMD --gradient_checkpointing"
fi

# 添加其他常用参数
TRAIN_CMD="$TRAIN_CMD --warmup_ratio 0.1"
TRAIN_CMD="$TRAIN_CMD --weight_decay 0.01"
TRAIN_CMD="$TRAIN_CMD --logging_steps 10"
TRAIN_CMD="$TRAIN_CMD --dataloader_num_workers 4"
TRAIN_CMD="$TRAIN_CMD --fp16"
TRAIN_CMD="$TRAIN_CMD --remove_unused_columns False"
TRAIN_CMD="$TRAIN_CMD --report_to tensorboard"

# 创建输出目录
OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

# 保存训练命令和环境信息
echo "$TRAIN_CMD" > "$OUTPUT_DIR/train_command.txt"
echo "训练时间: $(date)" > "$OUTPUT_DIR/training_info.txt"
echo "GPU配置: $GPU_IDS" >> "$OUTPUT_DIR/training_info.txt"
echo "训练阶段: $STAGE" >> "$OUTPUT_DIR/training_info.txt"
echo "实验名称: $EXPERIMENT_NAME" >> "$OUTPUT_DIR/training_info.txt"

# 显示最终命令
echo -e "${BLUE}即将执行的训练命令:${NC}"
echo -e "${GREEN}$TRAIN_CMD${NC}"
echo ""

# 询问确认
read -p "是否开始训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs${NC}"
echo ""

# 执行训练命令
eval $TRAIN_CMD

echo -e "${GREEN}训练完成！${NC}"
echo -e "${BLUE}结果保存在: $OUTPUT_DIR${NC}"