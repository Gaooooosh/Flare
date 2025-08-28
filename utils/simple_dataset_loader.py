#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据集加载器
专注核心功能，避免过度工程化
"""

import logging
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SimpleDatasetLoader:
    """简化的数据集加载器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
    
    def load_dataset(self, 
                    dataset_name: str, 
                    dataset_config: Optional[str] = None,
                    split: str = "train",
                    size_limit: Optional[int] = None) -> Dataset:
        """加载数据集"""
        logger.info(f"加载数据集: {dataset_name}")
        if dataset_config:
            logger.info(f"数据集配置: {dataset_config}")
        
        try:
            # 加载数据集
            if dataset_config:
                dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    cache_dir=self.cache_dir,
                    streaming=False
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    cache_dir=self.cache_dir,
                    streaming=False
                )
            
            # 限制大小
            if size_limit and len(dataset) > size_limit:
                dataset = dataset.select(range(size_limit))
                logger.info(f"限制数据集大小为: {size_limit}")
            
            logger.info(f"成功加载数据集，大小: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            # 返回简单的示例数据集
            return self._create_fallback_dataset(size_limit or 1000)
    
    def _create_fallback_dataset(self, size: int) -> Dataset:
        """创建回退数据集"""
        logger.warning(f"使用回退数据集，大小: {size}")
        
        sample_texts = [
            "这是一个示例文本，用于训练语言模型。",
            "自然语言处理是人工智能的重要分支。",
            "机器学习可以从数据中学习模式。",
            "深度学习使用神经网络进行学习。",
            "Transformer架构改变了NLP领域。",
        ]
        
        texts = []
        for i in range(size):
            base_text = sample_texts[i % len(sample_texts)]
            text = f"{base_text} 样本编号: {i+1}。"
            texts.append(text)
        
        return Dataset.from_dict({"text": texts})
    
    def tokenize_dataset(self, 
                        dataset: Dataset,
                        tokenizer: PreTrainedTokenizer,
                        text_column: str = "text",
                        max_length: int = 4096) -> Dataset:
        """对数据集进行分词"""
        logger.info(f"开始分词，最大长度: {max_length}")
        
        # 检查文本列
        if text_column not in dataset.column_names:
            # 尝试找到合适的文本列
            for col in ['content', 'document', 'article']:
                if col in dataset.column_names:
                    text_column = col
                    logger.info(f"使用文本列: {text_column}")
                    break
            else:
                raise ValueError(f"找不到文本列，可用列: {dataset.column_names}")
        
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding=False,
                max_length=max_length,
            )
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词处理",
        )
        
        logger.info(f"分词完成，数据集大小: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def split_dataset(self, 
                     dataset: Dataset, 
                     validation_split: float = 0.1) -> Tuple[Dataset, Optional[Dataset]]:
        """分割训练和验证集"""
        if validation_split <= 0:
            return dataset, None
        
        logger.info(f"分割数据集，验证集比例: {validation_split}")
        
        split_dataset = dataset.train_test_split(
            test_size=validation_split,
            seed=42
        )
        
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def prepare_dataset(self,
                       dataset_name: str,
                       tokenizer: PreTrainedTokenizer,
                       dataset_config: Optional[str] = None,
                       split: str = "train",
                       size_limit: Optional[int] = None,
                       validation_split: float = 0.1,
                       max_length: int = 4096,
                       text_column: str = "text") -> Tuple[Dataset, Optional[Dataset]]:
        """一站式数据集准备"""
        # 加载数据集
        dataset = self.load_dataset(dataset_name, dataset_config, split, size_limit)
        
        # 分词
        tokenized_dataset = self.tokenize_dataset(
            dataset, tokenizer, text_column, max_length
        )
        
        # 分割
        train_dataset, eval_dataset = self.split_dataset(
            tokenized_dataset, validation_split
        )
        
        return train_dataset, eval_dataset