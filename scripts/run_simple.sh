#!/bin/bash

# 简化的Qwen训练启动脚本
# 使用方法：
# ./run_simple.sh [stage] [gpu_ids] [experiment_name] [config_file]
# 例如：
#   ./run_simple.sh 1 "0,1" my_experiment
#   ./run_simple.sh 2 "0" debug_test simple_config.json

set -e

# 默认参数
STAGE=${1:-1}
GPU_IDS=${2:-"0"}
EXPERIMENT_NAME=${3:-"qwen_$(date +%Y%m%d_%H%M%S)"}
CONFIG_FILE=${4:-"simple_config.json"}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 简化的Qwen训练启动脚本 ===${NC}"
echo -e "${GREEN}训练阶段: ${STAGE}${NC}"
echo -e "${GREEN}使用GPU: ${GPU_IDS}${NC}"
echo -e "${GREEN}实验名称: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}配置文件: ${CONFIG_FILE}${NC}"
echo ""

# 参数验证
if [[ ! "$STAGE" =~ ^[12]$ ]]; then
    echo -e "${RED}错误: 训练阶段必须是 1 或 2${NC}"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查必要的文件
if [ ! -f "$PROJECT_ROOT/training/train_simple.py" ]; then
    echo -e "${RED}错误: 找不到训练脚本 train_simple.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 找不到配置文件 $CONFIG_FILE${NC}"
    exit 1
fi

# 设置GPU环境
if [ "$GPU_IDS" != "cpu" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo -e "${GREEN}设置GPU: $GPU_IDS${NC}"
else
    echo -e "${YELLOW}使用CPU模式${NC}"
fi

# 构建训练命令
TRAIN_CMD="python $PROJECT_ROOT/training/train_simple.py"
TRAIN_CMD="$TRAIN_CMD --config $PROJECT_ROOT/$CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --stage $STAGE"
TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"

# 显示命令
echo -e "${BLUE}执行命令:${NC}"
echo -e "${GREEN}$TRAIN_CMD${NC}"
echo ""

# 执行训练
echo -e "${GREEN}开始训练...${NC}"
eval $TRAIN_CMD

echo -e "${GREEN}训练完成！${NC}"