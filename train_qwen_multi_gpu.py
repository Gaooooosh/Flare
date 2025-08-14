#!/usr/bin/env python3
"""
多卡Qwen2.5-3B训练脚本
支持功能：
1. 多GPU训练（DDP/FSDP）
2. Flash-Attention 2
3. 长上下文（通过rope_scaling）
4. 指定层禁用RoPE
5. Hugging Face数据集集成
6. TensorBoard训练记录
7. 统一输出路径管理
8. 完善的评估脚本
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

# Hugging Face imports
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

# 导入自定义模块
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
        metadata={"help": "指定使用的GPU ID列表"}
    )
    gpu_type: Optional[str] = field(
        default=None,
        metadata={"help": "GPU类型偏好：A800, A40, 或None（自动选择）"}
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


class GPUManager:
    """GPU资源管理器"""
    
    @staticmethod
    def get_available_gpus() -> Dict[str, List[int]]:
        """获取可用GPU信息"""
        if not torch.cuda.is_available():
            return {"available": [], "A800": [], "A40": []}
        
        gpu_info = {"available": [], "A800": [], "A40": []}
        
        for i in range(torch.cuda.device_count()):
            try:
                # 检查GPU是否可用
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                
                # 获取GPU名称
                gpu_name = torch.cuda.get_device_name(i)
                gpu_info["available"].append(i)
                
                if "A800" in gpu_name:
                    gpu_info["A800"].append(i)
                elif "A40" in gpu_name:
                    gpu_info["A40"].append(i)
                    
            except Exception as e:
                logger.warning(f"GPU {i} 不可用: {e}")
        
        return gpu_info
    
    @staticmethod
    def select_gpus(gpu_type: Optional[str] = None, 
                   gpu_ids: Optional[List[int]] = None,
                   num_gpus: Optional[int] = None) -> List[int]:
        """选择GPU"""
        gpu_info = GPUManager.get_available_gpus()
        
        if gpu_ids:
            # 验证指定的GPU是否可用
            available_ids = [gid for gid in gpu_ids if gid in gpu_info["available"]]
            if not available_ids:
                raise ValueError(f"指定的GPU {gpu_ids} 都不可用")
            return available_ids
        
        # 根据GPU类型选择
        if gpu_type and gpu_type in gpu_info and gpu_info[gpu_type]:
            candidates = gpu_info[gpu_type]
        else:
            candidates = gpu_info["available"]
        
        if not candidates:
            raise ValueError("没有可用的GPU")
        
        # 限制GPU数量
        if num_gpus:
            candidates = candidates[:num_gpus]
        
        return candidates


class EarlyStoppingCallback(TrainerCallback):
    """早停回调"""
    
    def __init__(self, patience: int = 3, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.counter = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_metric = metrics.get("eval_loss", float("inf"))
        
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric > self.best_metric - self.threshold:
            self.counter += 1
            logger.info(f"早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info("触发早停")
                control.should_training_stop = True
        else:
            self.best_metric = current_metric
            self.counter = 0
        
        return control


class MemoryOptimizationCallback(TrainerCallback):
    """内存优化回调"""
    
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
    
    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


class PPLEvaluationCallback(TrainerCallback):
    """困惑度评估回调"""
    
    def __init__(self, eval_dataset, tokenizer, max_eval_samples: int = 1000):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_eval_samples = max_eval_samples
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        
        # 计算困惑度
        ppl = self.compute_perplexity(model)
        
        # 记录到TensorBoard
        if state.log_history:
            state.log_history[-1]["eval_perplexity"] = ppl
        
        logger.info(f"Perplexity: {ppl:.4f}")
    
    def compute_perplexity(self, model) -> float:
        """计算困惑度"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        # 选择评估样本
        eval_samples = min(len(self.eval_dataset), self.max_eval_samples)
        eval_data = self.eval_dataset.select(range(eval_samples))
        
        with torch.no_grad():
            for example in eval_data:
                input_ids = torch.tensor([example["input_ids"]], device=model.device)
                attention_mask = torch.tensor([example["attention_mask"]], device=model.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # 计算有效token数量
                valid_tokens = attention_mask.sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        model.train()
        return perplexity


def setup_output_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """设置输出目录"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"qwen_training_{timestamp}"
    
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "tensorboard").mkdir(exist_ok=True)
    (output_dir / "final_model").mkdir(exist_ok=True)
    
    return str(output_dir)


def setup_model_and_tokenizer(model_args: ModelArguments, 
                              selected_gpus: List[int]) -> tuple:
    """设置模型和分词器"""
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 数据类型映射
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping.get(model_args.torch_dtype, torch.bfloat16)
    
    # 模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "cache_dir": None
    }
    
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 多GPU设备映射
    if len(selected_gpus) > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"":selected_gpus[0]}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # 应用RoPE修改
    if model_args.no_rope_layers:
        patch_qwen_rope(model, no_rope_layers=model_args.no_rope_layers)
        logger.info(f"已禁用层 {model_args.no_rope_layers} 的RoPE")
    
    # 设置RoPE参数
    if hasattr(model.config, "rope_theta"):
        model.config.rope_theta = model_args.rope_theta
        logger.info(f"设置RoPE theta为: {model_args.rope_theta}")
    
    model.config.max_position_embeddings = model_args.max_position_embeddings
    model.config.nope_layers = model_args.no_rope_layers
    
    return model, tokenizer


def setup_datasets(data_args: DataArguments, tokenizer) -> Dict[str, Dataset]:
    """设置数据集"""
    logger.info(f"加载数据集: {data_args.dataset_name}")
    
    # 加载数据集
    load_kwargs = {
        "path": data_args.dataset_name,
        "split": data_args.dataset_split,
        "cache_dir": data_args.cache_dir
    }
    
    if data_args.dataset_config:
        load_kwargs["name"] = data_args.dataset_config
    
    try:
        raw_dataset = load_dataset(**load_kwargs)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        # 回退到本地数据集或示例数据
        logger.info("使用示例数据集")
        raw_dataset = Dataset.from_dict({
            "text": ["这是一个示例文本。" * 100] * 1000
        })
    
    # 限制数据集大小
    if data_args.dataset_size and len(raw_dataset) > data_args.dataset_size:
        raw_dataset = raw_dataset.select(range(data_args.dataset_size))
    
    # 分割训练集和验证集
    if data_args.validation_split_percentage > 0:
        split_dataset = raw_dataset.train_test_split(
            test_size=data_args.validation_split_percentage,
            seed=42
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = raw_dataset
        eval_dataset = None
    
    # 数据预处理
    def tokenize_function(examples):
        # 尝试不同的文本列名
        text_key = None
        for key in [data_args.text_column, "text", "content", "document", "raw_content"]:
            if key in examples:
                text_key = key
                break
        
        if text_key is None:
            raise ValueError(f"找不到文本列，可用列: {list(examples.keys())}")
        
        texts = examples[text_key]
        
        # 分词
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=data_args.max_seq_length,
            return_overflowing_tokens=False,
        )
        
        return tokenized
    
    # 应用分词
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="分词训练数据"
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="分词验证数据"
        )
        
        # 限制验证集大小以节省内存
        if len(eval_dataset) > 1000:
            eval_dataset = eval_dataset.select(range(1000))
    
    # 过滤空样本
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"验证集大小: {len(eval_dataset)}")
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset
    }


def set_freeze_layers(model, freeze: bool = True, no_rope_layers: List[int] = None):
    """冻结/解冻模型层"""
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = not freeze
    
    # 如果是冻结模式且指定了no_rope_layers，则解冻这些层
    if freeze and no_rope_layers:
        # 获取模型层
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            logger.warning("无法找到模型层，跳过层级冻结")
            return model
        
        for layer_idx in no_rope_layers:
            if 0 <= layer_idx < len(layers):
                logger.info(f"解冻层 {layer_idx}")
                for param in layers[layer_idx].parameters():
                    param.requires_grad = True
            else:
                logger.warning(f"层索引 {layer_idx} 超出范围，总层数 {len(layers)}")
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,}/{total_params:,} ({trainable_params/total_params:.2%})")
    
    return model


def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsExtended))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # GPU选择和设置
    selected_gpus = GPUManager.select_gpus(
        gpu_type=training_args.gpu_type,
        gpu_ids=training_args.gpu_ids
    )
    
    logger.info(f"选择的GPU: {selected_gpus}")
    
    # 设置CUDA设备
    if len(selected_gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpus[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    
    # 设置输出目录
    output_dir = setup_output_directory(
        training_args.base_output_dir,
        training_args.experiment_name
    )
    training_args.output_dir = str(Path(output_dir) / "checkpoints")
    
    if training_args.enable_tensorboard:
        training_args.logging_dir = str(Path(output_dir) / "tensorboard")
        training_args.report_to = ["tensorboard"]
    else:
        training_args.report_to = []
    
    logger.info(f"输出目录: {output_dir}")
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(model_args, selected_gpus)
    
    # 设置数据集
    datasets = setup_datasets(data_args, tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 根据训练阶段设置模型
    if training_args.stage == 1:
        logger.info("阶段1: 冻结预训练层，专攻新模块")
        model = set_freeze_layers(model, freeze=True, no_rope_layers=model_args.no_rope_layers)
    else:
        logger.info("阶段2: 解冻全模型，整体微调")
        model = set_freeze_layers(model, freeze=False)
    
    # 设置回调
    callbacks = [MemoryOptimizationCallback()]
    
    if training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=training_args.early_stopping_patience,
            threshold=training_args.early_stopping_threshold
        ))
    
    if datasets["eval_dataset"] is not None:
        callbacks.append(PPLEvaluationCallback(
            eval_dataset=datasets["eval_dataset"],
            tokenizer=tokenizer
        ))
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train_dataset"],
        eval_dataset=datasets["eval_dataset"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 检查是否有检查点可以恢复
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    if checkpoint is not None:
        logger.info(f"从检查点恢复训练: {checkpoint}")
    
    # 开始训练
    logger.info("开始训练...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存最终模型
    final_model_dir = str(Path(output_dir) / "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 保存训练指标
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练完成！模型保存在: {final_model_dir}")
    logger.info(f"训练指标保存在: {metrics_file}")


if __name__ == "__main__":
    main()