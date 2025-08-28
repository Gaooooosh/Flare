#!/bin/bash

# CUDA兼容性修复脚本
echo "正在尝试修复CUDA兼容性问题..."

# 设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.9/bin:$PATH

# 设置PyTorch CUDA设置
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export CUDA_LAUNCH_BLOCKING=1

# 显示当前设置
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 测试CUDA
echo "\n测试CUDA可用性..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA不可用，请检查驱动程序和PyTorch兼容性')
"

echo "\n如果CUDA仍然不可用，请尝试以下解决方案:"
echo "1. 重新安装兼容的PyTorch版本"
echo "2. 检查NVIDIA驱动程序版本"
echo "3. 重启系统"
echo "4. 联系系统管理员检查GPU权限"