#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的LoRA训练
"""

import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from patch_qwen_rope import patch_qwen_rope

def test_fixed_lora():
    """测试修复后的LoRA设置"""
    print("🧪 测试修复后的LoRA设置...")
    
    # 使用小模型
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # 加载模型
    print("📦 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 应用RoPE patch
    print("🔧 应用RoPE patch...")
    patch_qwen_rope(model, no_rope_layers=[0, 1])
    
    # 应用LoRA（不使用prepare_model_for_kbit_training）
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
    
    # 检查参数状态
    print("\n📊 检查参数状态:")
    trainable_count = 0
    total_count = 0
    lora_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        if param.requires_grad:
            trainable_count += 1
            if "lora_" in name:
                lora_count += 1
                print(f"✅ LoRA参数: {name}")
            else:
                print(f"⚠️ 非LoRA可训练参数: {name}")
    
    print(f"\n📈 统计:")
    print(f"总参数: {total_count}")
    print(f"可训练参数: {trainable_count}")
    print(f"LoRA参数: {lora_count}")
    
    # 测试前向传播和梯度
    print("\n🧪 测试前向传播和梯度...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        print(f"✅ 损失计算成功: {loss.item():.4f}")
        
        if loss.requires_grad:
            loss.backward()
            print("✅ 反向传播成功")
            
            # 检查梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                elif param.requires_grad:
                    print(f"⚠️ 参数需要梯度但没有梯度: {name}")
            
            print(f"📈 有梯度的参数: {grad_count}")
            return True
        else:
            print("❌ 损失不需要梯度！")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_lora()
    if success:
        print("\n🎉 修复后的LoRA测试通过！")
    else:
        print("\n💥 修复后的LoRA测试失败！")