#!/usr/bin/env python3
"""
Qwen大型语言模型多卡训练脚本 - 简化版
专注于核心训练功能，简化GPU配置
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import argparse
from datetime import datetime

# Transformers相关导入
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint

# 导入RoPE补丁
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from patch_qwen_rope import patch_qwen_rope

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B",
        metadata={"help": "预训练模型路径或Hugging Face模型名称"}
    )
    rope_theta: float = field(
        default=1000000.0,
        metadata={"help": "RoPE基频，用于长上下文扩展"}
    )
    max_position_embeddings: int = field(
        default=32768,
        metadata={"help": "最大位置嵌入长度"}
    )
    no_rope_layers: List[int] = field(
        default_factory=lambda: list(range(20, 33)),
        metadata={"help": "禁用RoPE的层号列表（0-based）"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "是否使用Flash Attention 2"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "模型数据类型：float16, bfloat16, float32"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    dataset_name: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        metadata={"help": "Hugging Face数据集名称"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称"}
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "数据集分割"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "文本列名称"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=16,
        metadata={"help": "数据预处理进程数"}
    )
    dataset_size: Optional[int] = field(
        default=100000,
        metadata={"help": "数据集大小限制，None表示使用全部数据"}
    )
    validation_split_percentage: float = field(
        default=0.1,
        metadata={"help": "验证集比例"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "数据集缓存目录"}
    )


@dataclass
class TrainingArgumentsExtended(TrainingArguments):
    """扩展的训练参数"""
    stage: int = field(
        default=1,
        metadata={"help": "训练阶段：1为冻结预训练层，2为全模型微调"}
    )
    gpu_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "指定使用的GPU ID列表，例如：[0,1,2,3]"}
    )
    base_output_dir: str = field(
        default="/work/xiaoyonggao",
        metadata={"help": "基础输出目录"}
    )
    experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "实验名称，用于创建子目录"}
    )
    enable_tensorboard: bool = field(
        default=True,
        metadata={"help": "是否启用TensorBoard记录"}
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "早停耐心值"}
    )
    early_stopping_threshold: float = field(
        default=0.001,
        metadata={"help": "早停阈值"}
    )


def setup_gpus(gpu_ids: Optional[List[int]] = None) -> List[int]:
    """设置GPU - 简化版"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，将使用CPU训练")
        return []
    
    total_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {total_gpus} 个GPU")
    
    # 显示所有GPU信息
    for i in range(total_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i}: {gpu_name}")
    
    # 如果指定了GPU ID
    if gpu_ids:
        # 验证GPU ID是否有效
        invalid_ids = [gid for gid in gpu_ids if gid >= total_gpus or gid < 0]
        if invalid_ids:
            raise ValueError(f"无效的GPU ID: {invalid_ids}，可用GPU ID: 0-{total_gpus-1}")
        
        selected_gpus = gpu_ids
        logger.info(f"使用指定的GPU: {selected_gpus}")
    else:
        # 使用所有可用GPU
        selected_gpus = list(range(total_gpus))
        logger.info(f"使用所有可用GPU: {selected_gpus}")
    
    # 设置CUDA可见设备
    if selected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        logger.info(f"设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    return selected_gpus


class EarlyStoppingCallback(TrainerCallback):
    """早停回调"""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_metric = metrics.get("eval_loss")
        if current_metric is None:
            return
        
        if self.best_metric is None or current_metric < self.best_metric - self.threshold:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"早停触发：{self.patience}次评估无改善")
                control.should_training_stop = True


class MemoryOptimizationCallback(TrainerCallback):
    """内存优化回调"""
    
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
    
    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


def setup_output_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """设置输出目录"""
    if experiment_name:
        output_dir = Path(base_dir) / experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(base_dir) / f"qwen_training_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    return str(output_dir)


def setup_model_and_tokenizer(model_args: ModelArguments) -> tuple:
    """设置模型和分词器"""
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    
    # 设置模型配置
    model_kwargs = {
        "torch_dtype": getattr(torch, model_args.torch_dtype),
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Flash Attention配置
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # 应用RoPE补丁
    logger.info(f"应用RoPE补丁，禁用层: {model_args.no_rope_layers}")
    patch_qwen_rope(
        model,
        rope_theta=model_args.rope_theta,
        max_position_embeddings=model_args.max_position_embeddings,
        no_rope_layers=model_args.no_rope_layers
    )
    
    return model, tokenizer


def setup_datasets(data_args: DataArguments, tokenizer) -> Dict[str, Dataset]:
    """设置数据集"""
    logger.info(f"加载数据集: {data_args.dataset_name}")
    
    # 加载数据集
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        split=data_args.dataset_split,
        cache_dir=data_args.cache_dir
    )
    
    # 限制数据集大小
    if data_args.dataset_size and len(dataset) > data_args.dataset_size:
        dataset = dataset.select(range(data_args.dataset_size))
        logger.info(f"限制数据集大小为: {data_args.dataset_size}")
    
    # 数据预处理函数
    def tokenize_function(examples):
        texts = examples[data_args.text_column]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=data_args.max_seq_length,
            return_overflowing_tokens=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 应用预处理
    logger.info("开始数据预处理...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # 分割训练集和验证集
    if data_args.validation_split_percentage > 0:
        split_dataset = tokenized_dataset.train_test_split(
            test_size=data_args.validation_split_percentage,
            seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"验证集大小: {len(eval_dataset)}")
    
    return {
        "train": train_dataset,
        "eval": eval_dataset
    }


def set_freeze_layers(model, freeze: bool = True, no_rope_layers: List[int] = None):
    """设置层冻结状态"""
    if no_rope_layers is None:
        no_rope_layers = []
    
    for name, param in model.named_parameters():
        if freeze:
            # 阶段1：冻结大部分参数，只训练特定层
            if any(f"layers.{layer}." in name for layer in no_rope_layers):
                param.requires_grad = True
                logger.info(f"解冻参数: {name}")
            else:
                param.requires_grad = False
        else:
            # 阶段2：解冻所有参数
            param.requires_grad = True
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


def main():
    """主函数"""
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsExtended))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置GPU
    selected_gpus = setup_gpus(training_args.gpu_ids)
    
    # 设置输出目录
    output_dir = setup_output_directory(
        training_args.base_output_dir,
        training_args.experiment_name
    )
    training_args.output_dir = output_dir
    
    # 设置TensorBoard
    if training_args.enable_tensorboard:
        training_args.logging_dir = os.path.join(output_dir, "logs")
        training_args.report_to = ["tensorboard"]
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 设置数据集
    datasets = setup_datasets(data_args, tokenizer)
    
    # 设置层冻结
    if training_args.stage == 1:
        logger.info("阶段1：冻结预训练层，专攻新模块")
        set_freeze_layers(model, freeze=True, no_rope_layers=model_args.no_rope_layers)
    else:
        logger.info("阶段2：解冻全模型，整体微调")
        set_freeze_layers(model, freeze=False)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 回调函数
    callbacks = [
        MemoryOptimizationCallback(),
    ]
    
    if datasets["eval"] is not None:
        callbacks.append(
            EarlyStoppingCallback(
                patience=training_args.early_stopping_patience,
                threshold=training_args.early_stopping_threshold
            )
        )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 检查断点续训
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint:
            logger.info(f"发现检查点，将从 {checkpoint} 继续训练")
    
    # 开始训练
    logger.info("开始训练...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    if datasets["eval"] is not None:
        logger.info("进行最终评估...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"训练完成！模型保存在: {training_args.output_dir}")


if __name__ == "__main__":
    main()