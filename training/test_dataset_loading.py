#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集加载测试脚本

此脚本用于测试数据集加载功能，验证HuggingFace数据集加载的兼容性和错误处理机制。
可以在不启动完整训练的情况下验证数据集加载是否正常工作。

使用方法:
    python test_dataset_loading.py --dataset_name "dataset_name" --text_column "text"
    python test_dataset_loading.py --config_file "path/to/config.json"
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from dataset_loader import DatasetLoader, DatasetLoadingError
from environment_adapter import EnvironmentAdapter
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_file: str) -> Dict[str, Any]:
    """从配置文件加载参数"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_file}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def test_tokenizer_loading(model_name: str) -> Optional[object]:
    """测试分词器加载"""
    try:
        logger.info(f"测试分词器加载: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("✓ 分词器加载成功")
        logger.info(f"  - 词汇表大小: {len(tokenizer)}")
        logger.info(f"  - PAD token: {tokenizer.pad_token}")
        logger.info(f"  - EOS token: {tokenizer.eos_token}")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"✗ 分词器加载失败: {e}")
        return None


def test_dataset_loading(dataset_name: str, 
                        dataset_config: Optional[str] = None,
                        dataset_split: str = "train",
                        text_column: str = "text",
                        max_samples: int = 1000,
                        cache_dir: Optional[str] = None,
                        use_cpu: bool = False) -> bool:
    """测试数据集加载"""
    try:
        logger.info(f"测试数据集加载: {dataset_name}")
        logger.info(f"  - 配置: {dataset_config}")
        logger.info(f"  - 分割: {dataset_split}")
        logger.info(f"  - 文本列: {text_column}")
        logger.info(f"  - 最大样本数: {max_samples}")
        logger.info(f"  - CPU模式: {use_cpu}")
        
        # 初始化数据集加载器
        dataset_loader = DatasetLoader(
            cache_dir=cache_dir,
            use_cpu=use_cpu
        )
        
        # 加载数据集
        dataset = dataset_loader.load_dataset_with_fallback(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            text_column=text_column,
            dataset_size=max_samples
        )
        
        if dataset is None:
            logger.error("✗ 数据集加载失败")
            return False
            
        logger.info("✓ 数据集加载成功")
        logger.info(f"  - 数据集大小: {len(dataset)}")
        logger.info(f"  - 列名: {dataset.column_names}")
        
        # 检查文本列
        if text_column in dataset.column_names:
            logger.info(f"✓ 找到文本列: {text_column}")
            
            # 显示前几个样本
            logger.info("前3个样本:")
            for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
                text = example[text_column]
                preview = text[:100] + "..." if len(text) > 100 else text
                logger.info(f"  样本 {i+1}: {preview}")
        else:
            logger.warning(f"✗ 未找到文本列: {text_column}")
            logger.info(f"可用列: {dataset.column_names}")
            
        return True
        
    except DatasetLoadingError as e:
        logger.error(f"✗ 数据集加载错误: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ 未知错误: {e}")
        return False


def test_dataset_preprocessing(dataset_name: str,
                             tokenizer,
                             text_column: str = "text",
                             max_seq_length: int = 512,
                             max_samples: int = 100,
                             cache_dir: Optional[str] = None,
                             use_cpu: bool = False) -> bool:
    """测试数据集预处理"""
    try:
        logger.info("测试数据集预处理...")
        
        # 初始化数据集加载器
        dataset_loader = DatasetLoader(
            cache_dir=cache_dir,
            use_cpu=use_cpu
        )
        
        # 加载数据集
        dataset = dataset_loader.load_dataset_with_fallback(
            dataset_name=dataset_name,
            text_column=text_column,
            dataset_size=max_samples
        )
        
        if dataset is None:
            logger.error("✗ 数据集加载失败，无法进行预处理测试")
            return False
            
        # 预处理数据集
        processed_dataset = dataset_loader.preprocess_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            max_seq_length=max_seq_length,
            num_workers=1  # 测试时使用单进程
        )
        
        if processed_dataset is None:
            logger.error("✗ 数据集预处理失败")
            return False
            
        logger.info("✓ 数据集预处理成功")
        logger.info(f"  - 预处理后大小: {len(processed_dataset)}")
        logger.info(f"  - 列名: {processed_dataset.column_names}")
        
        # 检查第一个样本
        if len(processed_dataset) > 0:
            sample = processed_dataset[0]
            logger.info(f"  - input_ids长度: {len(sample['input_ids'])}")
            logger.info(f"  - attention_mask长度: {len(sample['attention_mask'])}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据集预处理错误: {e}")
        return False


def test_environment_adapter() -> bool:
    """测试环境适配器"""
    try:
        logger.info("测试环境适配器...")
        
        # 初始化环境适配器
        env_adapter = EnvironmentAdapter()
        
        logger.info("✓ 环境适配器初始化成功")
        logger.info(f"  - 使用CPU: {env_adapter.env_info.use_cpu}")
        logger.info(f"  - CUDA可用: {env_adapter.env_info.cuda_available}")
        logger.info(f"  - GPU数量: {env_adapter.env_info.gpu_count}")
        logger.info(f"  - 总内存: {env_adapter.env_info.total_memory_gb:.1f} GB")
        logger.info(f"  - CPU核心数: {env_adapter.env_info.cpu_count}")
        
        # 测试推荐配置
        recommendations = env_adapter.get_recommended_config()
        logger.info("推荐配置:")
        logger.info(f"  - 批次大小: {recommendations['batch_size']}")
        logger.info(f"  - 工作进程数: {recommendations['num_workers']}")
        logger.info(f"  - 精度: {recommendations['precision']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 环境适配器测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集加载测试脚本")
    parser.add_argument("--config_file", type=str, help="配置文件路径")
    parser.add_argument("--dataset_name", type=str, default="togethercomputer/RedPajama-Data-1T-Sample", help="数据集名称")
    parser.add_argument("--dataset_config", type=str, help="数据集配置")
    parser.add_argument("--dataset_split", type=str, default="train", help="数据集分割")
    parser.add_argument("--text_column", type=str, default="text", help="文本列名称")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="模型名称")
    parser.add_argument("--max_samples", type=int, default=1000, help="最大样本数")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--cache_dir", type=str, help="缓存目录")
    parser.add_argument("--force_cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--skip_preprocessing", action="store_true", help="跳过预处理测试")
    
    args = parser.parse_args()
    
    # 从配置文件加载参数（如果提供）
    if args.config_file:
        try:
            config = load_config_from_file(args.config_file)
            
            # 从配置文件中提取数据参数
            data_args = config.get("data_args", {})
            model_args = config.get("model_args", {})
            
            args.dataset_name = data_args.get("dataset_name", args.dataset_name)
            args.dataset_config = data_args.get("dataset_config", args.dataset_config)
            args.dataset_split = data_args.get("dataset_split", args.dataset_split)
            args.text_column = data_args.get("text_column", args.text_column)
            args.model_name = model_args.get("model_name_or_path", args.model_name)
            args.max_seq_length = data_args.get("max_seq_length", args.max_seq_length)
            args.cache_dir = data_args.get("cache_dir", args.cache_dir)
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return 1
    
    logger.info("=" * 60)
    logger.info("数据集加载测试开始")
    logger.info("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    # 测试1: 环境适配器
    total_tests += 1
    logger.info("\n" + "=" * 40)
    logger.info("测试1: 环境适配器")
    logger.info("=" * 40)
    if test_environment_adapter():
        success_count += 1
    
    # 测试2: 分词器加载
    total_tests += 1
    logger.info("\n" + "=" * 40)
    logger.info("测试2: 分词器加载")
    logger.info("=" * 40)
    tokenizer = test_tokenizer_loading(args.model_name)
    if tokenizer is not None:
        success_count += 1
    
    # 测试3: 数据集加载
    total_tests += 1
    logger.info("\n" + "=" * 40)
    logger.info("测试3: 数据集加载")
    logger.info("=" * 40)
    if test_dataset_loading(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        use_cpu=args.force_cpu
    ):
        success_count += 1
    
    # 测试4: 数据集预处理（如果分词器加载成功且未跳过）
    if tokenizer is not None and not args.skip_preprocessing:
        total_tests += 1
        logger.info("\n" + "=" * 40)
        logger.info("测试4: 数据集预处理")
        logger.info("=" * 40)
        if test_dataset_preprocessing(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            text_column=args.text_column,
            max_seq_length=args.max_seq_length,
            max_samples=min(args.max_samples, 100),  # 预处理测试使用较少样本
            cache_dir=args.cache_dir,
            use_cpu=args.force_cpu
        ):
            success_count += 1
    
    # 输出测试结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    logger.info(f"总测试数: {total_tests}")
    logger.info(f"成功测试数: {success_count}")
    logger.info(f"失败测试数: {total_tests - success_count}")
    logger.info(f"成功率: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        logger.info("\n🎉 所有测试通过！数据集加载功能正常。")
        return 0
    else:
        logger.error(f"\n❌ {total_tests - success_count} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())