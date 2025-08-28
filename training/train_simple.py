#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的Qwen训练脚本
专注核心功能，避免过度工程化
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional, List

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


from config_manager import ConfigManager
from patch_qwen_rope import patch_qwen_rope
from environment_adapter import EnvironmentAdapter
from simple_dataset_loader import SimpleDatasetLoader

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config):
    """设置模型和分词器"""
    logger.info(f"加载模型: {config.model.model_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model_kwargs = {
        'torch_dtype': getattr(torch, config.model.torch_dtype),
        'device_map': 'auto' if not config.environment.force_cpu else None,
    }
    
    if config.model.use_flash_attention:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        **model_kwargs
    )
    
    # 应用RoPE补丁（在模型加载后）
    if config.model.no_rope_layers:
        patch_qwen_rope(model, config.model.no_rope_layers)
        logger.info(f"已禁用层 {config.model.no_rope_layers} 的RoPE")
    
    # 设置RoPE参数
    if hasattr(model.config, 'rope_theta'):
        model.config.rope_theta = config.model.rope_theta
        logger.info(f"设置rope_theta: {config.model.rope_theta}")
    
    return model, tokenizer


def setup_dataset(config, tokenizer):
    """设置数据集"""
    # 获取缓存目录
    cache_dir = getattr(config.data, 'cache_dir', None)
    dataset_loader = SimpleDatasetLoader(cache_dir=cache_dir)
    
    # 获取数据集配置
    dataset_config = getattr(config.data, 'dataset_config', None)
    
    # 使用简化的数据加载器
    train_dataset, eval_dataset = dataset_loader.prepare_dataset(
        dataset_name=config.data.dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        size_limit=config.data.dataset_size,
        validation_split=config.data.validation_split,
        max_length=config.model.max_seq_length,
    )
    
    return train_dataset, eval_dataset


def setup_training_args(config, output_dir):
    """设置训练参数"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # 训练设置
        do_train=True,
        do_eval=config.data.validation_split > 0,
        
        # 批次和轮数
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        num_train_epochs=config.training.num_epochs,
        
        # 学习率
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        
        # 评估和保存
        eval_strategy="steps" if config.data.validation_split > 0 else "no",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=3,
        
        # 日志
        logging_steps=config.training.logging_steps,
        report_to=[],  # 简化，不使用tensorboard
        
        # 其他
        seed=42,
        bf16=config.model.torch_dtype == "bfloat16",
        fp16=config.model.torch_dtype == "float16",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=config.data.validation_split > 0,
    )
    
    return training_args


def main():
    """主训练函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="简化的Qwen训练脚本")
    parser.add_argument("--config", type=str, default="simple_config.json", help="配置文件路径")
    parser.add_argument("--stage", type=int, default=1, help="训练阶段 (1或2)")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # 更新配置
    if args.stage:
        config.training.stage = args.stage
    if args.output_dir:
        config.output.base_dir = args.output_dir
    if args.experiment_name:
        config.output.experiment_name = args.experiment_name
    
    # 初始化环境适配器
    env_adapter = EnvironmentAdapter(force_cpu=config.environment.force_cpu)
    
    # 设置输出目录
    output_dir = Path(config.output.base_dir)
    if config.output.experiment_name:
        output_dir = output_dir / config.output.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"输出目录: {output_dir}")
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 设置数据集
    train_dataset, eval_dataset = setup_dataset(config, tokenizer)
    
    # 设置训练参数
    training_args = setup_training_args(config, str(output_dir))
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 根据训练阶段设置模型
    if config.training.stage == 1:
        logger.info("阶段1: 冻结预训练层")
        # 冻结除了最后几层的所有参数
        for name, param in model.named_parameters():
            if "layers.2" not in name and "layers.3" not in name and "lm_head" not in name:
                param.requires_grad = False
    else:
        logger.info("阶段2: 全模型微调")
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    logger.info(f"训练完成！模型保存在: {final_model_dir}")


if __name__ == "__main__":
    main()