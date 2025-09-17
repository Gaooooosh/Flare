#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA + RoPE 训练脚本使用示例
演示如何使用train_lora_rope.py进行模型训练
"""

import os
import sys
from pathlib import Path

def show_usage_example():
    """显示使用示例"""
    print("=" * 60)
    print("🚀 LoRA + RoPE 训练脚本使用示例")
    print("=" * 60)
    print()
    
    print("📋 1. 检查配置文件")
    config_file = Path("../simple_config.json")
    if config_file.exists():
        print(f"✅ 配置文件存在: {config_file.absolute()}")
        
        # 读取并显示关键配置
        sys.path.append(str(Path(__file__).parent.parent / 'utils'))
        from config_manager import ConfigManager
        
        config_manager = ConfigManager(str(config_file))
        config = config_manager.get_config()
        
        print(f"   - 模型: {config.model.model_name}")
        print(f"   - RoPE禁用层: {config.model.no_rope_layers}")
        print(f"   - 数据集: {config.data.dataset_name}")
        print(f"   - 批次大小: {config.training.batch_size}")
        print(f"   - 学习率: {config.training.learning_rate}")
        print(f"   - 训练轮数: {config.training.num_epochs}")
        print(f"   - GPU设备: {config.environment.gpu_ids}")
    else:
        print(f"⚠️ 配置文件不存在: {config_file.absolute()}")
        print("   请先运行交互式配置或手动创建配置文件")
    
    print()
    print("📋 2. 运行训练脚本")
    print("   命令: python train_lora_rope.py")
    print("   说明: 脚本会自动检测配置文件并启动训练")
    print()
    
    print("📋 3. 预期输出")
    print("   - 模型和分词器加载")
    print("   - RoPE层禁用确认")
    print("   - LoRA配置应用")
    print("   - 可训练参数统计")
    print("   - 训练进度显示")
    print("   - 模型保存到 /work/xiaoyonggao/{experiment_name}_lora/")
    print()
    
    print("📋 4. LoRA优势")
    print("   - 大幅减少可训练参数（通常<1%）")
    print("   - 显著降低显存需求")
    print("   - 加快训练速度")
    print("   - 保持模型性能")
    print()
    
    print("📋 5. 输出文件")
    print("   - final_lora_model/: LoRA权重和配置")
    print("   - adapter_config.json: LoRA配置文件")
    print("   - adapter_model.safetensors: LoRA权重文件")
    print("   - training_config.json: 训练配置备份")
    print()
    
    print("📋 6. 使用训练好的模型")
    print("   ```python")
    print("   from peft import PeftModel")
    print("   from transformers import AutoModelForCausalLM")
    print("   ")
    print("   # 加载基础模型")
    print("   base_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B')")
    print("   ")
    print("   # 加载LoRA权重")
    print("   model = PeftModel.from_pretrained(base_model, '/path/to/lora/model')")
    print("   ```")
    print()


def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        import peft
        print(f"✅ PEFT: {peft.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: 可用，{torch.cuda.device_count()} GPU")
        else:
            print("⚠️ CUDA: 不可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请安装缺失的依赖:")
        print("pip install torch transformers peft")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🧪 LoRA + RoPE 训练环境检查")
    print("=" * 60)
    print()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    print()
    
    # 显示使用示例
    show_usage_example()
    
    print("=" * 60)
    print("🎯 准备就绪！可以开始LoRA训练")
    print("=" * 60)
    print()
    print("下一步:")
    print("1. 确认配置文件设置正确")
    print("2. 运行: python train_lora_rope.py")
    print("3. 等待训练完成")
    print("4. 使用保存的LoRA模型进行推理")


if __name__ == "__main__":
    main()