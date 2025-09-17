#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度问题诊断脚本
"""

import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from config_manager import ConfigManager
from patch_qwen_rope import patch_qwen_rope

def debug_model_gradients():
    """调试模型梯度设置"""
    print("🔍 开始梯度诊断...")
    
    # 使用小模型进行测试
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # 加载模型
    print("📦 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("🔧 应用RoPE patch...")
    patch_qwen_rope(model, no_rope_layers=[0, 1])
    
    print("⚙️ 准备模型...")
    model = prepare_model_for_kbit_training(model)
    
    print("🎯 应用LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    print("\n📊 检查参数梯度设置:")
    trainable_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        if param.requires_grad:
            trainable_count += 1
            print(f"✅ {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        else:
            if "lora_" in name:
                print(f"❌ LoRA参数未设置梯度: {name}")
            # else:
            #     print(f"⚪ {name}: requires_grad={param.requires_grad}")
    
    print(f"\n📈 统计: {trainable_count}/{total_count} 参数可训练")
    
    if trainable_count == 0:
        print("❌ 没有可训练参数！")
        return False
    
    # 测试前向传播
    print("\n🧪 测试前向传播...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
    # 将输入移动到模型所在的设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ 前向传播成功")
        
        # 测试梯度计算
        print("\n🧪 测试梯度计算...")
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        print(f"✅ 损失计算成功: {loss.item():.4f}")
        
        # 检查损失是否需要梯度
        print(f"🔍 损失梯度设置: requires_grad={loss.requires_grad}")
        
        if loss.requires_grad:
            loss.backward()
            print("✅ 反向传播成功")
            
            # 检查哪些参数有梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                elif param.requires_grad:
                    print(f"⚠️ 参数需要梯度但没有梯度: {name}")
            
            print(f"📈 有梯度的参数: {grad_count}")
            
        else:
            print("❌ 损失不需要梯度！")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = debug_model_gradients()
    if success:
        print("\n🎉 梯度诊断通过！")
    else:
        print("\n💥 梯度诊断失败！")