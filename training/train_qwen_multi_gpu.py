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
import dataclasses
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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from patch_qwen_rope import patch_qwen_rope
from dataset_loader import DatasetLoader, DatasetLoadingError
from environment_adapter import EnvironmentAdapter
from memory_optimizer import MemoryOptimizer
from .error_handler import ErrorHandler, TrainingError, DatasetError, ModelError, EnvironmentError, MemoryError, error_handler_decorator, safe_execute, get_global_error_handler

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
    force_cpu: bool = field(
        default=False,
        metadata={"help": "强制使用CPU模式，即使有GPU可用"}
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
    
    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
    
    def on_evaluate(self, args, state, control, **kwargs):
        self.memory_optimizer.cleanup_memory()
    
    def on_save(self, args, state, control, **kwargs):
        self.memory_optimizer.cleanup_memory()
    
    def on_step_end(self, args, state, control, **kwargs):
        # 每100步执行一次内存清理
        if state.global_step % 100 == 0:
            self.memory_optimizer.cleanup_memory()


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


@error_handler_decorator(get_global_error_handler(), reraise=True)
def setup_model_and_tokenizer(model_args: ModelArguments, 
                              selected_gpus: List[int]) -> tuple:
    """设置模型和分词器"""
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    
    # 检查是否为CPU模式
    use_cpu = len(selected_gpus) == 0 or model_args.force_cpu
    
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
    
    # CPU模式下强制使用float32
    if use_cpu:
        torch_dtype = torch.float32
        logger.info("CPU模式：使用float32数据类型")
    
    # 模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "cache_dir": None
    }
    
    # Flash Attention配置（仅在GPU模式下启用）
    if model_args.use_flash_attention and not use_cpu:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # 设备映射
    if use_cpu:
        model_kwargs["device_map"] = None
    elif len(selected_gpus) > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"":selected_gpus[0]}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    
    # CPU模式下移动模型到CPU
    if use_cpu:
        model = model.to('cpu')
        logger.info("模型已移动到CPU")
    
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


@error_handler_decorator(get_global_error_handler(), reraise=True)
def setup_datasets(data_args: DataArguments, tokenizer, use_cpu=False) -> Dict[str, Dataset]:
    """设置数据集（使用新的DatasetLoader）"""
    logger.info(f"使用DatasetLoader加载数据集: {data_args.dataset_name}")
    
    try:
        # 创建数据集加载器
        dataset_loader = DatasetLoader(
            cache_dir=data_args.cache_dir,
            use_cpu=use_cpu
        )
        
        # 加载数据集
        dataset = dataset_loader.load_dataset_with_fallback(
            dataset_name=data_args.dataset_name,
            dataset_config=data_args.dataset_config,
            dataset_split=data_args.dataset_split,
            text_column=data_args.text_column,
            max_samples=data_args.dataset_size
        )
        
        # 分割数据集
        train_dataset, eval_dataset = dataset_loader.split_dataset(
            dataset=dataset,
            validation_split_percentage=data_args.validation_split_percentage * 100,
            seed=42
        )
        
        # 预处理数据集
        train_dataset = dataset_loader.preprocess_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            text_column=data_args.text_column,
            max_seq_length=data_args.max_seq_length,
            num_proc=data_args.preprocessing_num_workers,
            description="分词训练数据"
        )
        
        if eval_dataset is not None:
            eval_dataset = dataset_loader.preprocess_dataset(
                dataset=eval_dataset,
                tokenizer=tokenizer,
                text_column=data_args.text_column,
                max_seq_length=data_args.max_seq_length,
                num_proc=data_args.preprocessing_num_workers,
                description="分词验证数据"
            )
            
            # 限制验证集大小以节省内存
            if len(eval_dataset) > 1000:
                eval_dataset = eval_dataset.select(range(1000))
        
        logger.info(f"数据集设置完成 - 训练集: {len(train_dataset)}, 验证集: {len(eval_dataset) if eval_dataset else 0}")
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }
        
    except DatasetLoadingError as e:
        logger.error(f"数据集加载失败: {e}")
        raise
    except Exception as e:
        logger.error(f"数据集设置过程中发生未知错误: {e}")
        raise DatasetLoadingError(f"数据集设置失败: {e}") from e


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
    
    # 检查是否有配置文件参数
    import sys
    config_file = None
    remaining_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--config_file" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            i += 2  # 跳过config_file参数和其值
        else:
            remaining_args.append(sys.argv[i])
            i += 1
    
    if config_file and os.path.exists(config_file):
        logger.info(f"从配置文件加载参数: {config_file}")
        # 先从配置文件加载
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 分别解析各个部分
        model_args = ModelArguments(**config_data.get('model_args', {}))
        data_args = DataArguments(**config_data.get('data_args', {}))
        
        # 处理training_args，转换旧的参数名
        training_args_data = config_data.get('training_args', {})
        if 'evaluation_strategy' in training_args_data:
            training_args_data['eval_strategy'] = training_args_data.pop('evaluation_strategy')
        
        training_args = TrainingArgumentsExtended(**training_args_data)
        
        # 如果有额外的命令行参数，用它们覆盖配置文件参数
        if len(remaining_args) > 1:  # remaining_args[0]是脚本名
            logger.info("使用命令行参数覆盖配置文件参数")
            # 临时修改sys.argv以便解析剩余参数
            original_argv = sys.argv
            sys.argv = remaining_args
            try:
                model_args_override, data_args_override, training_args_override = parser.parse_args_into_dataclasses()
                # 合并参数（命令行参数优先）
                for field in model_args.__dataclass_fields__:
                    if hasattr(model_args_override, field):
                        override_value = getattr(model_args_override, field)
                        field_info = model_args.__dataclass_fields__[field]
                        
                        # 处理default_factory字段
                        if field_info.default_factory is not dataclasses.MISSING:
                            default_value = field_info.default_factory()
                        else:
                            default_value = field_info.default
                        
                        # 只有当命令行值与默认值不同时才覆盖
                        if override_value != default_value:
                            setattr(model_args, field, override_value)
                
                for field in data_args.__dataclass_fields__:
                    if hasattr(data_args_override, field):
                        override_value = getattr(data_args_override, field)
                        field_info = data_args.__dataclass_fields__[field]
                        
                        # 处理default_factory字段
                        if field_info.default_factory is not dataclasses.MISSING:
                            default_value = field_info.default_factory()
                        else:
                            default_value = field_info.default
                        
                        # 只有当命令行值与默认值不同时才覆盖
                        if override_value != default_value:
                            setattr(data_args, field, override_value)
                
                for field in training_args.__dataclass_fields__:
                    if hasattr(training_args_override, field):
                        override_value = getattr(training_args_override, field)
                        field_info = training_args.__dataclass_fields__[field]
                        
                        # 处理default_factory字段
                        if field_info.default_factory is not dataclasses.MISSING:
                            default_value = field_info.default_factory()
                        else:
                            default_value = field_info.default
                        
                        # 只有当命令行值与默认值不同时才覆盖
                        if override_value != default_value:
                            setattr(training_args, field, override_value)
            finally:
                sys.argv = original_argv
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 初始化环境适配器
    force_cpu = getattr(model_args, 'force_cpu', False)
    env_adapter = EnvironmentAdapter(force_cpu=force_cpu)
    
    # 初始化内存优化器
    memory_optimizer = MemoryOptimizer(use_cpu=env_adapter.env_info.use_cpu)
    memory_optimizations = memory_optimizer.optimize_for_training()
    logger.info(f"内存优化建议: {memory_optimizations}")
    
    # 根据环境和内存优化调整训练参数
    training_args = env_adapter.adapt_training_args(training_args)
    
    # 应用内存优化建议
    if 'recommended_batch_size' in memory_optimizations:
        training_args.per_device_train_batch_size = memory_optimizations['recommended_batch_size']
        training_args.per_device_eval_batch_size = memory_optimizations['recommended_batch_size']
    if 'gradient_accumulation_steps' in memory_optimizations:
        training_args.gradient_accumulation_steps = memory_optimizations['gradient_accumulation_steps']
    if 'dataloader_num_workers' in memory_optimizations:
        training_args.dataloader_num_workers = memory_optimizations['dataloader_num_workers']
    
    # GPU选择和设置（仅在非CPU模式下）
    if not env_adapter.env_info.use_cpu:
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
    else:
        selected_gpus = []
        logger.info("CPU模式：跳过GPU选择")
    
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
    
    # 初始化错误处理器
    error_log_file = output_dir / "error.log"
    error_handler = ErrorHandler(log_file=str(error_log_file), enable_recovery=True)
    
    # 设置为全局错误处理器
    from .error_handler import set_global_error_handler
    set_global_error_handler(error_handler)
    
    logger.info(f"错误处理器已初始化，日志文件: {error_log_file}")
    
    # 根据环境调整模型配置
    model_args = env_adapter.adapt_model_config(model_args)
    
    # 设置模型和分词器
    logger.info("设置模型和分词器...")
    try:
        model, tokenizer = setup_model_and_tokenizer(model_args, selected_gpus)
    except Exception as e:
        model_error = ModelError(
            f"模型设置失败: {str(e)}",
            model_name=model_args.model_name_or_path,
            context={'selected_gpus': selected_gpus}
        )
        if not error_handler.handle_error(model_error):
            logger.critical("模型设置失败，无法继续训练")
            raise
    
    # 设置数据集
    logger.info("设置数据集...")
    try:
        datasets = setup_datasets(data_args, tokenizer, use_cpu=env_adapter.env_info.use_cpu)
    except DatasetLoadingError as e:
        dataset_error = DatasetError(
            f"数据集设置失败: {str(e)}",
            dataset_name=data_args.dataset_name,
            context={'use_cpu': env_adapter.env_info.use_cpu}
        )
        if not error_handler.handle_error(dataset_error):
            logger.critical("数据集设置失败，无法继续训练")
            raise
    except Exception as e:
        dataset_error = DatasetError(
            f"数据集设置过程中发生未知错误: {str(e)}",
            dataset_name=data_args.dataset_name,
            context={'use_cpu': env_adapter.env_info.use_cpu}
        )
        if not error_handler.handle_error(dataset_error):
            logger.critical("数据集设置失败，无法继续训练")
            raise
    
    # 如果没有评估数据集，禁用 load_best_model_at_end 以避免冲突
    if datasets["eval_dataset"] is None:
        logger.warning("没有评估数据集，禁用 load_best_model_at_end")
        training_args.load_best_model_at_end = False
        training_args.eval_strategy = "no"
    
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
    callbacks = [MemoryOptimizationCallback(memory_optimizer)]
    
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
    
    # 启动内存监控
    memory_optimizer.start_monitoring()
    
    # 开始训练
    logger.info("开始训练...")
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    finally:
        # 停止内存监控
        memory_optimizer.stop_monitoring()
    
    # 保存最终模型
    final_model_dir = str(Path(output_dir) / "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 保存训练指标
    metrics_file = Path(output_dir) / "training_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2, ensure_ascii=False)
    
    # 保存内存使用报告
    memory_optimizer.save_memory_report(output_dir)
    
    # 最终内存清理
    memory_optimizer.cleanup_memory()
    
    # 保存错误报告
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        error_handler.save_error_report(Path(output_dir) / "error_report.json")
        logger.info(f"训练过程中发生了 {error_summary['total_errors']} 个错误，详细信息已保存到错误报告")
    
    logger.info(f"训练完成！模型保存在: {final_model_dir}")
    logger.info(f"训练指标保存在: {metrics_file}")
    logger.info(f"内存报告保存在: {Path(output_dir) / 'memory_report.json'}")
    if error_summary['total_errors'] > 0:
        logger.info(f"错误报告保存在: {Path(output_dir) / 'error_report.json'}")


if __name__ == "__main__":
    main()