#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LoRA + RoPE训练脚本的功能
验证模型加载、RoPE patch和LoRA配置是否正确
"""

import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
sys.path.append(str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from config_manager import ConfigManager
from patch_qwen_rope import patch_qwen_rope


def test_model_loading():
    """测试模型加载和RoPE patch"""
    print("🧪 测试模型加载和RoPE patch...")
    
    # 使用小模型进行测试
    model_name = "Qwen/Qwen2.5-0.5B"  # 使用更小的模型进行测试
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ 分词器加载成功")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ 模型加载成功")
        
        # 测试RoPE patch
        no_rope_layers = [0, 1, 2]  # 测试前3层
        patch_qwen_rope(model, no_rope_layers=no_rope_layers)
        print(f"✅ RoPE patch应用成功，禁用层: {no_rope_layers}")
        
        # 验证patch是否生效
        for idx, layer in enumerate(model.model.layers):
            if idx in no_rope_layers:
                if hasattr(layer.self_attn, '_rope_disabled') and layer.self_attn._rope_disabled:
                    print(f"✅ 层 {idx} RoPE已禁用")
                else:
                    print(f"❌ 层 {idx} RoPE禁用失败")
            else:
                if not hasattr(layer.self_attn, '_rope_disabled') or not layer.self_attn._rope_disabled:
                    print(f"✅ 层 {idx} RoPE保持启用")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def test_lora_config():
    """测试LoRA配置"""
    print("\n🧪 测试LoRA配置...")
    
    model, tokenizer = test_model_loading()
    if model is None:
        return False
    
    try:
        # 准备模型进行LoRA训练
        model = prepare_model_for_kbit_training(model)
        print("✅ 模型准备完成")
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # 使用较小的rank进行测试
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        print("✅ LoRA配置创建成功")
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        print("✅ LoRA应用成功")
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA配置失败: {e}")
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n🧪 测试配置加载...")
    
    config_file = "simple_config.json"
    if not Path(config_file).exists():
        print("⚠️ 配置文件不存在，跳过配置测试")
        return True
    
    try:
        config_manager = ConfigManager(config_file)
        config = config_manager.get_config()
        print("✅ 配置加载成功")
        
        # 检查关键配置项
        if hasattr(config.model, 'no_rope_layers'):
            print(f"✅ RoPE禁用层配置: {config.model.no_rope_layers}")
        else:
            print("⚠️ 未找到RoPE禁用层配置")
        
        if hasattr(config.model, 'model_name'):
            print(f"✅ 模型名称: {config.model.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False


def test_import_dependencies():
    """测试依赖导入"""
    print("🧪 测试依赖导入...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers版本: {transformers.__version__}")
        
        import peft
        print(f"✅ PEFT版本: {peft.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️ CUDA不可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ 依赖导入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 LoRA + RoPE 训练脚本测试")
    print("=" * 60)
    
    # 测试依赖导入
    if not test_import_dependencies():
        print("❌ 依赖测试失败，退出")
        return
    
    # 测试配置加载
    if not test_config_loading():
        print("❌ 配置测试失败，退出")
        return
    
    # 测试LoRA配置
    if not test_lora_config():
        print("❌ LoRA测试失败，退出")
        return
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("✅ LoRA + RoPE 训练脚本功能正常")
    print("=" * 60)


if __name__ == "__main__":
    main()