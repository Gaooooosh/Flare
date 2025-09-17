#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整的LoRA训练流程
"""

import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

def create_dummy_dataset(tokenizer, size=100):
    """创建虚拟数据集用于测试"""
    texts = [f"This is test sentence number {i}." for i in range(size)]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False, 
            max_length=512
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def test_training_flow():
    """测试完整的训练流程"""
    print("🧪 测试LoRA训练流程...")
    
    # 使用小模型
    model_name = "Qwen/Qwen2.5-0.5B"
    
    # 加载分词器
    print("📦 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("📦 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 准备模型
    print("⚙️ 准备模型...")
    model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA
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
    
    # 验证梯度
    print("🔍 验证梯度设置...")
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    print(f"可训练参数: {trainable_count}/{total_count}")
    
    if trainable_count == 0:
        print("❌ 没有可训练参数！")
        return False
    
    # 创建数据集
    print("📊 创建测试数据集...")
    train_dataset = create_dummy_dataset(tokenizer, 50)
    eval_dataset = create_dummy_dataset(tokenizer, 10)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 训练参数
    print("⚙️ 设置训练参数...")
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        remove_unused_columns=True,
        save_only_model=True,
        max_steps=10,  # 只训练10步用于测试
    )
    
    # 创建训练器
    print("🏗️ 创建训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 测试训练
    print("🎯 开始测试训练...")
    try:
        trainer.train()
        print("✅ 训练测试成功！")
        return True
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_flow()
    if success:
        print("\n🎉 LoRA训练流程测试通过！")
    else:
        print("\n💥 LoRA训练流程测试失败！")