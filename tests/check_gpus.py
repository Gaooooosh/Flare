#!/usr/bin/env python3
"""
GPU检查工具
用于查看当前服务器的GPU配置和可用性
"""

import torch
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
    from train_qwen_multi_gpu import GPUManager
except ImportError:
    print("❌ 无法导入GPUManager，请确保train_qwen_multi_gpu.py存在")
    sys.exit(1)


def print_separator(title="", char="=", width=60):
    """打印分隔线"""
    if title:
        title = f" {title} "
        padding = (width - len(title)) // 2
        line = char * padding + title + char * (width - padding - len(title))
    else:
        line = char * width
    print(line)


def check_cuda_environment():
    """检查CUDA环境"""
    print_separator("CUDA环境检查")
    
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print(f"🔥 PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.version.cuda}")
        print(f"🎯 cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print("❌ CUDA不可用")
        return False
    
    return True


def check_gpu_details():
    """检查GPU详细信息"""
    print_separator("GPU详细信息")
    
    if not torch.cuda.is_available():
        print("❌ 没有可用的GPU")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"🔢 GPU总数: {gpu_count}")
    print()
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / (1024**3)  # GB
        
        # 安全地获取内存使用情况
        try:
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_free = memory_total - memory_reserved
        except Exception:
            # 如果无法获取内存信息，使用总内存作为可用内存
            memory_allocated = 0.0
            memory_reserved = 0.0
            memory_free = memory_total
        
        print(f"📱 GPU {i}: {props.name}")
        print(f"   💾 总内存: {memory_total:.1f} GB")
        print(f"   🟢 可用内存: {memory_free:.1f} GB")
        print(f"   🔴 已用内存: {memory_reserved:.1f} GB")
        print(f"   ⚡ 计算能力: {props.major}.{props.minor}")
        print(f"   🔧 多处理器: {props.multi_processor_count}")
        print()


def check_gpu_manager():
    """检查GPU管理器功能"""
    print_separator("GPU管理器检查")
    
    try:
        # 获取GPU信息
        gpu_info = GPUManager.get_available_gpus()
        
        print(f"🎯 可用GPU: {gpu_info['available']}")
        
        if 'A800' in gpu_info and gpu_info['A800']:
            print(f"🚀 A800 GPU: {gpu_info['A800']}")
        
        if 'A40' in gpu_info and gpu_info['A40']:
            print(f"⚡ A40 GPU: {gpu_info['A40']}")
        
        if 'other' in gpu_info and gpu_info['other']:
            print(f"🔧 其他GPU: {gpu_info['other']}")
        
        print()
        
        # 测试GPU选择功能
        print("🧪 测试GPU选择功能:")
        
        # 自动选择
        try:
            auto_gpus = GPUManager.select_gpus()
            print(f"   自动选择: {auto_gpus}")
        except Exception as e:
            print(f"   自动选择失败: {e}")
        
        # 按类型选择A800
        try:
            a800_gpus = GPUManager.select_gpus(gpu_type="A800")
            print(f"   A800选择: {a800_gpus}")
        except Exception as e:
            print(f"   A800选择失败: {e}")
        
        # 按类型选择A40
        try:
            a40_gpus = GPUManager.select_gpus(gpu_type="A40")
            print(f"   A40选择: {a40_gpus}")
        except Exception as e:
            print(f"   A40选择失败: {e}")
        
        # 手动选择
        try:
            manual_gpus = GPUManager.select_gpus(gpu_ids=[0, 1])
            print(f"   手动选择[0,1]: {manual_gpus}")
        except Exception as e:
            print(f"   手动选择失败: {e}")
            
    except Exception as e:
        print(f"❌ GPU管理器检查失败: {e}")


def show_usage_examples():
    """显示使用示例"""
    print_separator("使用示例")
    
    print("🚀 启动脚本方式:")
    print("   # 自动选择GPU")
    print("   ./run_training.sh 1 auto my_experiment")
    print()
    print("   # 指定GPU类型")
    print("   ./run_training.sh 1 A800 my_experiment")
    print()
    print("   # 手动选择GPU")
    print("   ./run_training.sh 1 manual my_experiment \"0,1,2,3\"")
    print()
    
    print("🐍 Python脚本方式:")
    print("   # 指定GPU ID")
    print("   python train_qwen_multi_gpu.py --gpu_ids 0 1 2 3")
    print()
    print("   # 指定GPU类型")
    print("   python train_qwen_multi_gpu.py --gpu_type A800")
    print()
    print("   # 单GPU训练")
    print("   python train_qwen_multi_gpu.py --gpu_ids 0")
    print()


def show_recommendations():
    """显示推荐配置"""
    print_separator("推荐配置")
    
    if not torch.cuda.is_available():
        print("❌ 无GPU可用，无法提供推荐")
        return
    
    try:
        gpu_info = GPUManager.get_available_gpus()
        
        print("💡 根据当前GPU配置的推荐:")
        print()
        
        if 'A800' in gpu_info and gpu_info['A800']:
            print(f"🚀 大模型训练 (推荐A800): GPU {gpu_info['A800']}")
            print("   ./run_training.sh 1 A800 large_model_experiment")
            print()
        
        if 'A40' in gpu_info and gpu_info['A40']:
            print(f"⚡ 标准训练 (推荐A40): GPU {gpu_info['A40']}")
            print("   ./run_training.sh 1 A40 standard_experiment")
            print()
        
        if len(gpu_info['available']) >= 4:
            print("🔥 多GPU并行训练:")
            first_four = gpu_info['available'][:4]
            print(f"   ./run_training.sh 1 manual multi_gpu_experiment \"{','.join(map(str, first_four))}\"")
            print()
        
        print("🧪 调试/测试 (单GPU):")
        print(f"   ./run_training.sh 1 manual debug_experiment \"{gpu_info['available'][0]}\"")
        print()
        
    except Exception as e:
        print(f"❌ 无法生成推荐: {e}")


def main():
    """主函数"""
    print("🔍 GPU检查工具")
    print_separator()
    
    # 检查CUDA环境
    if not check_cuda_environment():
        print("\n❌ CUDA环境不可用，无法继续检查")
        return 1
    
    print()
    
    # 检查GPU详细信息
    check_gpu_details()
    
    # 检查GPU管理器
    check_gpu_manager()
    
    print()
    
    # 显示使用示例
    show_usage_examples()
    
    # 显示推荐配置
    show_recommendations()
    
    print_separator("检查完成")
    print("✅ GPU检查完成！")
    print("📖 更多信息请查看: README_MIGRATION.md")
    print("📋 配置示例请查看: training_config_gpu_examples.json")
    
    return 0


if __name__ == "__main__":
    exit(main())