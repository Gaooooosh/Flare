#!/usr/bin/env python3
"""
数据集加载模块
提供健壮的HuggingFace数据集加载功能，支持多种数据格式和错误恢复机制
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DatasetLoadingError(Exception):
    """数据集加载异常"""
    pass


class DatasetLoader:
    """
    数据集加载器
    提供健壮的数据集加载、预处理和错误恢复功能
    """
    
    def __init__(self, cache_dir: Optional[str] = None, use_cpu: bool = False):
        """
        初始化数据集加载器
        
        Args:
            cache_dir: 数据集缓存目录
            use_cpu: 是否使用CPU模式（影响内存优化策略）
        """
        self.cache_dir = cache_dir
        self.use_cpu = use_cpu
        self._setup_cache_dir()
    
    def _setup_cache_dir(self) -> None:
        """设置缓存目录"""
        if self.cache_dir is None:
            # 默认缓存目录
            self.cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"数据集缓存目录: {self.cache_dir}")
    
    def _get_fallback_dataset(self, size: int = 1000) -> Dataset:
        """
        获取回退数据集（当主数据集加载失败时使用）
        
        Args:
            size: 数据集大小
            
        Returns:
            Dataset: 示例数据集
        """
        logger.warning(f"使用回退数据集，大小: {size}")
        
        # 创建多样化的示例文本
        sample_texts = [
            "这是一个关于自然语言处理的示例文本。自然语言处理是人工智能的重要分支。",
            "机器学习算法可以从数据中学习模式，并做出预测或决策。",
            "深度学习使用神经网络来模拟人脑的学习过程。",
            "Transformer架构彻底改变了自然语言处理领域。",
            "预训练语言模型在各种NLP任务中都表现出色。",
            "大型语言模型具有强大的文本生成和理解能力。",
            "人工智能技术正在快速发展，应用领域越来越广泛。",
            "数据是机器学习的燃料，高质量的数据至关重要。",
        ]
        
        # 扩展到指定大小
        texts = []
        for i in range(size):
            base_text = sample_texts[i % len(sample_texts)]
            # 添加一些变化
            extended_text = f"{base_text} 这是第{i+1}个样本。" + " 额外的内容用于增加文本长度。" * (i % 5 + 1)
            texts.append(extended_text)
        
        return Dataset.from_dict({"text": texts})
    
    def _detect_text_column(self, dataset: Dataset, preferred_column: str = "text") -> str:
        """
        检测数据集中的文本列
        
        Args:
            dataset: 数据集
            preferred_column: 首选列名
            
        Returns:
            str: 文本列名
            
        Raises:
            DatasetLoadingError: 找不到合适的文本列时抛出
        """
        available_columns = list(dataset.column_names)
        logger.info(f"数据集可用列: {available_columns}")
        
        # 按优先级检查可能的文本列名
        candidate_columns = [
            preferred_column,
            "text", "content", "document", "raw_content", 
            "input", "sentence", "passage", "article",
            "body", "description", "summary"
        ]
        
        for col in candidate_columns:
            if col in available_columns:
                logger.info(f"使用文本列: {col}")
                return col
        
        # 如果没有找到标准列名，尝试第一个字符串类型的列
        for col in available_columns:
            try:
                sample_value = dataset[0][col]
                if isinstance(sample_value, str) and len(sample_value) > 10:
                    logger.warning(f"自动检测到文本列: {col}")
                    return col
            except (IndexError, KeyError, TypeError):
                continue
        
        raise DatasetLoadingError(f"无法找到合适的文本列。可用列: {available_columns}")
    
    def load_dataset_with_fallback(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        dataset_split: str = "train",
        dataset_size: Optional[int] = None,
        text_column: str = "text"
    ) -> Dataset:
        """
        加载数据集，支持多种错误恢复策略
        
        Args:
            dataset_name: 数据集名称
            dataset_config: 数据集配置
            dataset_split: 数据集分割
            dataset_size: 数据集大小限制
            text_column: 文本列名
            
        Returns:
            Dataset: 加载的数据集
        """
        logger.info(f"开始加载数据集: {dataset_name}")
        
        # 构建加载参数
        load_kwargs = {
            "path": dataset_name,
            "split": dataset_split,
            "cache_dir": self.cache_dir
        }
        
        if dataset_config:
            load_kwargs["name"] = dataset_config
        
        # 尝试加载数据集
        dataset = None
        error_messages = []
        
        # 策略1: 直接加载
        try:
            logger.info("尝试直接加载数据集...")
            dataset = load_dataset(**load_kwargs)
            logger.info(f"成功加载数据集，大小: {len(dataset)}")
        except Exception as e:
            error_msg = f"直接加载失败: {str(e)}"
            logger.warning(error_msg)
            error_messages.append(error_msg)
        
        # 策略2: 尝试不同的分割
        if dataset is None and dataset_split != "train":
            try:
                logger.info("尝试使用train分割...")
                load_kwargs["split"] = "train"
                dataset = load_dataset(**load_kwargs)
                logger.info(f"使用train分割成功，大小: {len(dataset)}")
            except Exception as e:
                error_msg = f"train分割加载失败: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # 策略3: 尝试加载整个数据集然后选择分割
        if dataset is None:
            try:
                logger.info("尝试加载完整数据集...")
                load_kwargs_full = load_kwargs.copy()
                load_kwargs_full.pop("split", None)
                full_dataset = load_dataset(**load_kwargs_full)
                
                if isinstance(full_dataset, DatasetDict):
                    # 尝试获取指定分割
                    if dataset_split in full_dataset:
                        dataset = full_dataset[dataset_split]
                    elif "train" in full_dataset:
                        dataset = full_dataset["train"]
                        logger.warning(f"未找到{dataset_split}分割，使用train分割")
                    else:
                        # 使用第一个可用分割
                        first_split = list(full_dataset.keys())[0]
                        dataset = full_dataset[first_split]
                        logger.warning(f"使用第一个可用分割: {first_split}")
                else:
                    dataset = full_dataset
                
                logger.info(f"完整数据集加载成功，大小: {len(dataset)}")
            except Exception as e:
                error_msg = f"完整数据集加载失败: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
        
        # 策略4: 使用回退数据集
        if dataset is None:
            logger.error("所有数据集加载策略都失败，使用回退数据集")
            for error_msg in error_messages:
                logger.error(f"  - {error_msg}")
            
            fallback_size = dataset_size if dataset_size else 1000
            dataset = self._get_fallback_dataset(fallback_size)
        
        # 限制数据集大小
        if dataset_size and len(dataset) > dataset_size:
            logger.info(f"限制数据集大小从 {len(dataset)} 到 {dataset_size}")
            dataset = dataset.select(range(dataset_size))
        
        # 验证文本列
        try:
            actual_text_column = self._detect_text_column(dataset, text_column)
            if actual_text_column != text_column:
                logger.info(f"文本列从 '{text_column}' 更改为 '{actual_text_column}'")
        except DatasetLoadingError as e:
            logger.error(f"文本列检测失败: {e}")
            raise
        
        return dataset
    
    def preprocess_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        max_seq_length: int = 4096,
        num_workers: int = None
    ) -> Dataset:
        """
        预处理数据集
        
        Args:
            dataset: 原始数据集
            tokenizer: 分词器
            text_column: 文本列名
            max_seq_length: 最大序列长度
            num_workers: 处理进程数
            
        Returns:
            Dataset: 预处理后的数据集
        """
        logger.info(f"开始预处理数据集，大小: {len(dataset)}")
        
        # 自动检测文本列
        actual_text_column = self._detect_text_column(dataset, text_column)
        
        # 根据CPU模式调整处理参数
        if num_workers is None:
            if self.use_cpu:
                # CPU模式下使用较少的进程数
                num_workers = min(4, os.cpu_count() or 1)
            else:
                # GPU模式下可以使用更多进程数
                num_workers = min(16, os.cpu_count() or 1)
        
        logger.info(f"使用 {num_workers} 个进程进行数据预处理")
        
        def tokenize_function(examples):
            """分词函数"""
            texts = examples[actual_text_column]
            
            # 确保文本是字符串列表
            if isinstance(texts, str):
                texts = [texts]
            
            # 过滤空文本
            texts = [text if isinstance(text, str) and text.strip() else "空文本" for text in texts]
            
            # 分词
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_attention_mask=True,
            )
            
            return tokenized
        
        # 应用分词
        try:
            processed_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=num_workers,
                remove_columns=dataset.column_names,
                desc="分词处理",
                load_from_cache_file=True,  # 启用缓存
            )
        except Exception as e:
            logger.warning(f"批量处理失败，尝试单进程处理: {e}")
            processed_dataset = dataset.map(
                tokenize_function,
                batched=True,
                num_proc=1,
                remove_columns=dataset.column_names,
                desc="分词处理（单进程）",
            )
        
        # 过滤空样本
        original_size = len(processed_dataset)
        processed_dataset = processed_dataset.filter(
            lambda x: len(x["input_ids"]) > 0 and sum(x["attention_mask"]) > 0
        )
        filtered_size = len(processed_dataset)
        
        if filtered_size < original_size:
            logger.warning(f"过滤了 {original_size - filtered_size} 个空样本")
        
        logger.info(f"数据预处理完成，最终大小: {filtered_size}")
        return processed_dataset
    
    def split_dataset(
        self,
        dataset: Dataset,
        validation_split_percentage: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Optional[Dataset]]:
        """
        分割数据集为训练集和验证集
        
        Args:
            dataset: 原始数据集
            validation_split_percentage: 验证集比例
            seed: 随机种子
            
        Returns:
            Dict: 包含train_dataset和eval_dataset的字典
        """
        if validation_split_percentage <= 0 or validation_split_percentage >= 1:
            logger.info("不分割验证集")
            return {
                "train_dataset": dataset,
                "eval_dataset": None
            }
        
        logger.info(f"分割数据集，验证集比例: {validation_split_percentage}")
        
        split_dataset = dataset.train_test_split(
            test_size=validation_split_percentage,
            seed=seed
        )
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        # 限制验证集大小以节省内存（特别是在CPU模式下）
        max_eval_size = 1000 if self.use_cpu else 2000
        if len(eval_dataset) > max_eval_size:
            logger.info(f"限制验证集大小从 {len(eval_dataset)} 到 {max_eval_size}")
            eval_dataset = eval_dataset.select(range(max_eval_size))
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(eval_dataset)}")
        
        return {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset
        }