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
import random
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
        num_workers: int = None,
        dynamic_length: bool = True,
        segmentation_strategy: str = "none",
        segment_stride: Optional[int] = None,
        mixed_seq_lengths: Optional[List[int]] = None,
        mixed_seq_probs: Optional[List[float]] = None,
    ) -> Dataset:
        """
        预处理数据集
        
        Args:
            dataset: 原始数据集
            tokenizer: 分词器
            text_column: 文本列名
            max_seq_length: 最大序列长度
            num_workers: 处理进程数
            dynamic_length: 是否启用动态长度（不在预处理阶段padding）
            segmentation_strategy: 长文本分段策略：none, chunk, sliding_window, mixed
            segment_stride: 滑动窗口步长（token数）
            mixed_seq_lengths: 混合长度列表
            mixed_seq_probs: 混合长度采样概率
            
        Returns:
            Dataset: 预处理后的数据集（variable-length，包含length列）
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
        
        logger.info(f"使用 {num_workers} 个进程进行数据预处理 | segmentation={segmentation_strategy} | dynamic_length={dynamic_length}")
        
        # 归一化混合长度参数
        if mixed_seq_lengths:
            mixed_seq_lengths = [int(l) for l in mixed_seq_lengths if l and l > 0]
            if not mixed_seq_lengths:
                mixed_seq_lengths = None
        if mixed_seq_lengths and mixed_seq_probs:
            if len(mixed_seq_probs) != len(mixed_seq_lengths):
                logger.warning("mixed_seq_probs 与 mixed_seq_lengths 长度不一致，将使用均匀分布")
                mixed_seq_probs = None
            else:
                total_p = sum(mixed_seq_probs)
                if total_p <= 0:
                    mixed_seq_probs = None
                else:
                    mixed_seq_probs = [p/total_p for p in mixed_seq_probs]
        
        if segmentation_strategy not in {"none", "chunk", "sliding_window", "mixed"}:
            logger.warning(f"未知的分段策略 {segmentation_strategy}，将使用 none")
            segmentation_strategy = "none"
        
        if segmentation_strategy == "sliding_window" and (segment_stride is None or segment_stride <= 0):
            segment_stride = max(1, max_seq_length // 4)
            logger.info(f"自动设置滑动窗口步长: {segment_stride}")
        
        eos_id = tokenizer.eos_token_id
        
        def segment_ids(ids: List[int]) -> List[List[int]]:
            L = len(ids)
            segments: List[List[int]] = []
            if L == 0:
                return segments
            
            if segmentation_strategy == "none":
                # 仅截断到max_seq_length
                seg = ids[:max_seq_length]
                segments.append(seg)
                return segments
            
            if segmentation_strategy == "chunk":
                for i in range(0, L, max_seq_length):
                    seg = ids[i:i+max_seq_length]
                    if len(seg) > 0:
                        segments.append(seg)
                return segments
            
            if segmentation_strategy == "sliding_window":
                step = segment_stride or max(1, max_seq_length // 4)
                for i in range(0, max(1, L - max_seq_length + 1), step):
                    seg = ids[i:i+max_seq_length]
                    if len(seg) > 0:
                        segments.append(seg)
                # 如果最后一段不足且没有覆盖到末尾，补充最后一段
                if L > 0 and (not segments or segments[-1][-1] != ids[-1]):
                    last_start = max(0, L - max_seq_length)
                    seg = ids[last_start: last_start + max_seq_length]
                    if len(seg) > 0:
                        segments.append(seg)
                return segments
            
            if segmentation_strategy == "mixed" and mixed_seq_lengths:
                i = 0
                while i < L:
                    if mixed_seq_probs:
                        length_choice = random.choices(mixed_seq_lengths, weights=mixed_seq_probs, k=1)[0]
                    else:
                        length_choice = random.choice(mixed_seq_lengths)
                    length_choice = max(1, min(length_choice, max_seq_length))
                    seg = ids[i:i+length_choice]
                    if len(seg) == 0:
                        break
                    segments.append(seg)
                    i += length_choice
                return segments
            
            # 默认回退为chunk
            for i in range(0, L, max_seq_length):
                seg = ids[i:i+max_seq_length]
                if len(seg) > 0:
                    segments.append(seg)
            return segments
        
        def tokenize_and_segment(examples: Dict[str, Any]) -> Dict[str, List[List[int]]]:
            texts = examples[actual_text_column]
            if isinstance(texts, str):
                texts = [texts]
            # 清洗文本
            texts = [t if isinstance(t, str) and t.strip() else "空文本" for t in texts]
            input_ids_list: List[List[int]] = []
            attention_masks_list: List[List[int]] = []
            lengths: List[int] = []
            
            for t in texts:
                # 不在这里padding与截断（除了分段策略本身的切分）
                ids = tokenizer.encode(t, add_special_tokens=False)
                if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
                    # 追加eos，利于语言建模
                    ids = ids + [eos_id]
                segs = segment_ids(ids)
                for seg in segs:
                    input_ids_list.append(seg)
                    attention_masks_list.append([1] * len(seg))
                    lengths.append(len(seg))
            return {
                "input_ids": input_ids_list,
                "attention_mask": attention_masks_list,
                "length": lengths,
            }
        
        # 应用分词与分段
        try:
            processed_dataset = dataset.map(
                tokenize_and_segment,
                batched=True,
                num_proc=num_workers,
                remove_columns=dataset.column_names,
                desc="分词与分段处理",
                load_from_cache_file=True,
            )
        except Exception as e:
            logger.warning(f"批量多进程处理失败，尝试单进程: {e}")
            processed_dataset = dataset.map(
                tokenize_and_segment,
                batched=True,
                num_proc=1,
                remove_columns=dataset.column_names,
                desc="分词与分段处理（单进程）",
            )
        
        # 过滤空样本与过长样本（安全检查）
        def valid_example(ex):
            ids = ex.get("input_ids", [])
            L = len(ids)
            if L == 0:
                return False
            if L > max_seq_length:
                # 极端情况下mixed策略可能超界，这里再次裁剪
                ex["input_ids"] = ids[:max_seq_length]
                ex["attention_mask"] = [1] * len(ex["input_ids"])
                ex["length"] = len(ex["input_ids"])
            return True
        
        original_size = len(processed_dataset)
        processed_dataset = processed_dataset.filter(valid_example)
        filtered_size = len(processed_dataset)
        if filtered_size < original_size:
            logger.info(f"过滤了 {original_size - filtered_size} 个无效样本")
        
        logger.info(f"数据预处理完成，最终大小: {filtered_size}")
        return processed_dataset