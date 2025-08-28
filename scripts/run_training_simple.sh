#!/bin/bash

# Qwen多卡训练启动脚本 - 简化版
# 使用方法：
# ./run_training_simple.sh [stage] [gpu_ids] [experiment_name]
# 例如：
#   ./run_training_simple.sh 1 "0,1,2,3" my_experiment
#   ./run_training_simple.sh 2 "0,1" debug_test
#   ./run_training_simple.sh 1 "0" single_gpu_test
# 环境变量：
#   AUTO_CONFIRM=yes    # 跳过确认提示，直接开始训练（用于CI/非交互环境）
#   SMOKE_TEST=1        # 启用快速冒烟测试参数（极少步数与小数据量）

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

# 检查uv环境
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: 找不到uv命令${NC}"
    echo -e "${YELLOW}请确保已安装uv依赖管理工具${NC}"
    exit 1
fi

# 设置Python命令使用uv run
PYTHON_CMD="uv run python"

echo -e "${GREEN}使用Python: uv run python${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查必要的文件
if [ ! -f "$PROJECT_ROOT/training/train_simple.py" ]; then
    echo -e "${RED}错误: 找不到训练脚本 train_simple.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/utils/patch_qwen_rope.py" ]; then
    echo -e "${RED}错误: 找不到RoPE补丁文件 patch_qwen_rope.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/simple_config.json" ]; then
    echo -e "${RED}错误: 找不到配置文件 simple_config.json${NC}"
    exit 1
fi

# 显示GPU信息
echo -e "${YELLOW}检查GPU环境...${NC}"
$PYTHON_CMD - << 'PYCODE'
import torch
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: 没有检测到CUDA GPU')
PYCODE

# 验证指定的GPU是否存在
echo -e "${YELLOW}验证指定的GPU...${NC}"
$PYTHON_CMD - << PYCODE
import sys, torch
gpu_ids = [int(x) for x in "$GPU_IDS".split(',')]
available_gpus = list(range(torch.cuda.device_count()))
for gpu_id in gpu_ids:
    if gpu_id not in available_gpus:
        print(f'错误: GPU {gpu_id} 不存在')
        sys.exit(1)
print(f'✅ 指定的GPU {gpu_ids} 都可用')
PYCODE

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU验证失败，请检查GPU ID${NC}"
    exit 1
fi

# 构建训练命令
TRAIN_CMD="$PYTHON_CMD $PROJECT_ROOT/training/train_simple.py"

# 添加基本参数
TRAIN_CMD="$TRAIN_CMD --config $PROJECT_ROOT/simple_config.json"
TRAIN_CMD="$TRAIN_CMD --stage $STAGE"
TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# 根据阶段显示信息
if [ "$STAGE" = "1" ]; then
    echo -e "${YELLOW}阶段1: 冻结预训练层，专攻新模块${NC}"
elif [ "$STAGE" = "2" ]; then
    echo -e "${YELLOW}阶段2: 解冻全模型，整体微调${NC}"
fi

# SMOKE测试模式提示
if [ -n "$SMOKE_TEST" ] && [[ "$SMOKE_TEST" =~ ^(1|y|Y|yes|YES|true|TRUE)$ ]]; then
    echo -e "${YELLOW}启用SMOKE_TEST模式：使用极少数据与步数进行快速冒烟测试${NC}"
fi

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

# 是否自动确认
AUTO_CONFIRM_FLAG=${AUTO_CONFIRM:-}
if [ -n "$AUTO_CONFIRM_FLAG" ] && [[ "$AUTO_CONFIRM_FLAG" =~ ^(1|y|Y|yes|YES|true|TRUE)$ ]]; then
    REPLY="y"
else
    # 询问确认
    read -p "是否开始训练? (y/N): " -n 1 -r
    echo
fi

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    exit 0
fi

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
if [ -n "$SMOKE_TEST" ] && [[ "$SMOKE_TEST" =~ ^(1|y|Y|yes|YES|true|TRUE)$ ]]; then
    echo -e "${YELLOW}当前为冒烟测试模式，训练过程会非常简短${NC}"
fi
echo -e "${YELLOW}输出目录: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs${NC}"
echo ""

# 执行训练命令
eval $TRAIN_CMD

echo -e "${GREEN}训练完成！${NC}"
echo -e "${BLUE}结果保存在: $OUTPUT_DIR${NC}"