#!/bin/bash

# Qwen多卡训练启动脚本
# 使用方法：
# ./run_training.sh [stage] [gpu_type] [experiment_name] [gpu_ids]
# 例如：
#   ./run_training.sh 1 A800 qwen_experiment_001
#   ./run_training.sh 1 auto qwen_experiment_001 "0,1,2,3"
#   ./run_training.sh 2 manual qwen_experiment_001 "4,5,6,7"

set -e

# 默认参数
STAGE=${1:-1}
GPU_TYPE=${2:-"auto"}
EXPERIMENT_NAME=${3:-"qwen_training_$(date +%Y%m%d_%H%M%S)"}
GPU_IDS=${4:-""}
BASE_OUTPUT_DIR="/work/xiaoyonggao"
# CONFIG_FILE将在脚本中动态设置

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Qwen多卡训练启动脚本 ===${NC}"
echo -e "${GREEN}训练阶段: ${STAGE}${NC}"
echo -e "${GREEN}GPU类型: ${GPU_TYPE}${NC}"
if [ -n "$GPU_IDS" ]; then
    echo -e "${GREEN}指定GPU: ${GPU_IDS}${NC}"
fi
echo -e "${GREEN}实验名称: ${EXPERIMENT_NAME}${NC}"
echo -e "${GREEN}输出目录: ${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}${NC}"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 找不到Python解释器${NC}"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查必要的文件
if [ ! -f "$PROJECT_ROOT/training/train_qwen_multi_gpu.py" ]; then
    echo -e "${RED}错误: 找不到训练脚本 train_qwen_multi_gpu.py${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/utils/patch_qwen_rope.py" ]; then
    echo -e "${RED}错误: 找不到模型修改脚本 patch_qwen_rope.py${NC}"
    exit 1
fi

# 检查配置文件
if [ ! -f "$PROJECT_ROOT/configs/training_config.json" ]; then
    echo -e "${YELLOW}警告: 找不到配置文件 training_config.json，将使用默认参数${NC}"
fi

# 创建输出目录
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}创建输出目录: $OUTPUT_DIR${NC}"

# 检查GPU可用性
echo -e "${BLUE}检查GPU可用性...${NC}"
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: 没有检测到CUDA GPU')
"

# 构建训练命令
TRAIN_CMD="python $PROJECT_ROOT/training/train_qwen_multi_gpu.py"

# 检查是否使用配置文件
CONFIG_FILE="$PROJECT_ROOT/configs/training_config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}使用配置文件: $CONFIG_FILE${NC}"
    TRAIN_CMD="$TRAIN_CMD --config_file $CONFIG_FILE"
else
    echo -e "${YELLOW}配置文件不存在，使用命令行参数${NC}"
fi

# 添加基本参数（这些参数会覆盖配置文件中的对应参数）
TRAIN_CMD="$TRAIN_CMD --stage $STAGE"
TRAIN_CMD="$TRAIN_CMD --base_output_dir $BASE_OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"
TRAIN_CMD="$TRAIN_CMD --enable_tensorboard"

# 添加GPU类型参数
if [ "$GPU_TYPE" != "auto" ] && [ "$GPU_TYPE" != "manual" ]; then
    TRAIN_CMD="$TRAIN_CMD --gpu_type $GPU_TYPE"
fi

# 添加GPU ID参数
if [ -n "$GPU_IDS" ]; then
    # 将逗号分隔的字符串转换为Python列表格式
    GPU_LIST=$(echo "$GPU_IDS" | sed 's/,/ /g')
    TRAIN_CMD="$TRAIN_CMD --gpu_ids $GPU_LIST"
    echo -e "${YELLOW}使用指定GPU: $GPU_IDS${NC}"
fi

# 根据阶段调整参数
if [ "$STAGE" = "1" ]; then
    echo -e "${YELLOW}阶段1: 冻结预训练层，专攻新模块${NC}"
    TRAIN_CMD="$TRAIN_CMD --learning_rate 1e-4"
    TRAIN_CMD="$TRAIN_CMD --per_device_train_batch_size 3"
    TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps 16"
    TRAIN_CMD="$TRAIN_CMD --max_steps 1000"
    TRAIN_CMD="$TRAIN_CMD --eval_steps 50"
    TRAIN_CMD="$TRAIN_CMD --save_steps 100"
elif [ "$STAGE" = "2" ]; then
    echo -e "${YELLOW}阶段2: 解冻全模型，整体微调${NC}"
    TRAIN_CMD="$TRAIN_CMD --learning_rate 3e-5"
    TRAIN_CMD="$TRAIN_CMD --per_device_train_batch_size 2"
    TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps 16"
    TRAIN_CMD="$TRAIN_CMD --max_steps 2000"
    TRAIN_CMD="$TRAIN_CMD --eval_steps 20"
    TRAIN_CMD="$TRAIN_CMD --save_steps 50"
    TRAIN_CMD="$TRAIN_CMD --gradient_checkpointing"
else
    echo -e "${RED}错误: 不支持的训练阶段 $STAGE (支持: 1, 2)${NC}"
    exit 1
fi

# 添加其他常用参数
TRAIN_CMD="$TRAIN_CMD --bf16"
TRAIN_CMD="$TRAIN_CMD --dataloader_num_workers 4"
TRAIN_CMD="$TRAIN_CMD --logging_steps 10"
TRAIN_CMD="$TRAIN_CMD --warmup_ratio 0.1"
TRAIN_CMD="$TRAIN_CMD --weight_decay 0.01"
TRAIN_CMD="$TRAIN_CMD --save_total_limit 3"
# 注意：evaluation_strategy和load_best_model_at_end将由训练脚本根据是否有验证集自动设置
# 如果有验证集，会自动启用evaluation_strategy=steps和load_best_model_at_end=True
# 如果没有验证集，会自动设置evaluation_strategy=no和load_best_model_at_end=False
TRAIN_CMD="$TRAIN_CMD --metric_for_best_model eval_loss"
TRAIN_CMD="$TRAIN_CMD --early_stopping_patience 3"

# 保存训练命令到文件
echo "$TRAIN_CMD" > "${OUTPUT_DIR}/train_command.txt"
echo -e "${GREEN}训练命令已保存到: ${OUTPUT_DIR}/train_command.txt${NC}"

# 保存环境信息
echo "=== 环境信息 ===" > "${OUTPUT_DIR}/environment_info.txt"
echo "时间: $(date)" >> "${OUTPUT_DIR}/environment_info.txt"
echo "用户: $(whoami)" >> "${OUTPUT_DIR}/environment_info.txt"
echo "主机: $(hostname)" >> "${OUTPUT_DIR}/environment_info.txt"
echo "Python版本: $(python --version)" >> "${OUTPUT_DIR}/environment_info.txt"
echo "工作目录: $(pwd)" >> "${OUTPUT_DIR}/environment_info.txt"
echo "训练命令: $TRAIN_CMD" >> "${OUTPUT_DIR}/environment_info.txt"
echo "" >> "${OUTPUT_DIR}/environment_info.txt"
echo "=== GPU信息 ===" >> "${OUTPUT_DIR}/environment_info.txt"
python -c "
import torch
print(f'CUDA版本: {torch.version.cuda}')
print(f'PyTorch版本: {torch.__version__}')
print(f'GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" >> "${OUTPUT_DIR}/environment_info.txt" 2>/dev/null || echo "无法获取GPU信息" >> "${OUTPUT_DIR}/environment_info.txt"

# 询问是否开始训练
echo ""
echo -e "${YELLOW}即将执行训练命令:${NC}"
echo -e "${BLUE}$TRAIN_CMD${NC}"
echo ""
read -p "是否开始训练? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}训练已取消${NC}"
    echo -e "${GREEN}您可以稍后手动执行以下命令开始训练:${NC}"
    echo -e "${BLUE}$TRAIN_CMD${NC}"
    exit 0
fi

# 开始训练
echo -e "${GREEN}开始训练...${NC}"
echo -e "${YELLOW}训练日志将保存到: ${OUTPUT_DIR}/training.log${NC}"
echo -e "${YELLOW}TensorBoard日志目录: ${OUTPUT_DIR}/tensorboard${NC}"
echo -e "${YELLOW}可以使用以下命令查看TensorBoard:${NC}"
echo -e "${BLUE}tensorboard --logdir=${OUTPUT_DIR}/tensorboard${NC}"
echo ""

# 执行训练（同时输出到控制台和日志文件）
$TRAIN_CMD 2>&1 | tee "${OUTPUT_DIR}/training.log"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}训练完成！${NC}"
    echo -e "${GREEN}模型保存在: ${OUTPUT_DIR}/final_model${NC}"
    echo -e "${GREEN}检查点保存在: ${OUTPUT_DIR}/checkpoints${NC}"
    echo -e "${GREEN}训练日志: ${OUTPUT_DIR}/training.log${NC}"
    echo -e "${GREEN}TensorBoard日志: ${OUTPUT_DIR}/tensorboard${NC}"
else
    echo -e "${RED}训练失败，请检查日志文件: ${OUTPUT_DIR}/training.log${NC}"
    exit 1
fi

# 询问是否运行评估
echo ""
read -p "是否运行模型评估? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}开始模型评估...${NC}"
    EVAL_CMD="python $PROJECT_ROOT/utils/evaluate_model_enhanced.py"
    EVAL_CMD="$EVAL_CMD --model_path ${OUTPUT_DIR}/final_model"
    EVAL_CMD="$EVAL_CMD --output_dir ${OUTPUT_DIR}/evaluation"
    EVAL_CMD="$EVAL_CMD --use_default_datasets"
    EVAL_CMD="$EVAL_CMD --max_samples_per_dataset 1000"
    
    echo -e "${BLUE}评估命令: $EVAL_CMD${NC}"
    $EVAL_CMD 2>&1 | tee "${OUTPUT_DIR}/evaluation.log"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}评估完成！结果保存在: ${OUTPUT_DIR}/evaluation${NC}"
    else
        echo -e "${RED}评估失败，请检查日志: ${OUTPUT_DIR}/evaluation.log${NC}"
    fi
fi

echo -e "${GREEN}所有任务完成！${NC}"