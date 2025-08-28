#!/usr/bin/env python3
"""
测试可以正常加载的大型文本数据集
"""

import sys
import os
sys.path.append('/home/xiaoyonggao/Flare')

from utils.simple_dataset_loader import SimpleDatasetLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset(dataset_name, dataset_config=None, size_limit=100):
    """测试单个数据集的加载"""
    
    cache_dir = "/datacache/huggingface"
    loader = SimpleDatasetLoader(cache_dir=cache_dir)
    
    print(f"\n{'='*60}")
    print(f"测试数据集: {dataset_name}")
    if dataset_config:
        print(f"配置: {dataset_config}")
    print(f"{'='*60}")
    
    try:
        dataset = loader.load_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="train",
            size_limit=size_limit
        )
        
        print(f"✅ 数据集加载成功!")
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集列名: {dataset.column_names}")
        
        # 查看第一个样本
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"第一个样本的键: {list(first_sample.keys())}")
            
            # 查找文本列
            text_column = None
            for col in ['text', 'content', 'document', 'raw_content', 'article']:
                if col in first_sample:
                    text_column = col
                    break
            
            if text_column:
                text_preview = first_sample[text_column][:200] + "..." if len(first_sample[text_column]) > 200 else first_sample[text_column]
                print(f"文本列 '{text_column}' 预览:")
                print(f"'{text_preview}'")
            else:
                print(f"第一个样本内容: {first_sample}")
        
        return True, dataset_name, dataset_config
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False, dataset_name, dataset_config

def main():
    """测试多个数据集"""
    
    # 测试数据集列表 - 这些都是使用 Parquet 格式的数据集
    test_datasets = [
        # C4 数据集 - 大型网络爬取数据集
        ("allenai/c4", "en", 50),  # 英文子集，限制50个样本
        
        # OpenWebText - GPT-2 训练数据的开源版本
        ("Skylion007/openwebtext", None, 100),
        
        # BookCorpus - 书籍文本数据集
        ("bookcorpus", None, 100),
        
        # WikiText - 维基百科文本
        ("wikitext", "wikitext-2-raw-v1", 100),
        ("wikitext", "wikitext-103-raw-v1", 100),
        
        # Common Crawl 新闻数据集
        ("cc_news", None, 100),
        
        # 多语言 Common Crawl
        ("mc4", "en", 50),
        
        # 英文新闻数据集
        ("cnn_dailymail", "3.0.0", 50),
        
        # 学术论文摘要
        ("scientific_papers", "arxiv", 50),
    ]
    
    successful_datasets = []
    failed_datasets = []
    
    print("🔍 开始测试可用的大型文本数据集...")
    
    for dataset_name, dataset_config, size_limit in test_datasets:
        success, name, config = test_dataset(dataset_name, dataset_config, size_limit)
        if success:
            successful_datasets.append((name, config))
        else:
            failed_datasets.append((name, config))
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print(f"{'='*60}")
    
    print(f"\n✅ 成功加载的数据集 ({len(successful_datasets)}):")
    for name, config in successful_datasets:
        if config:
            print(f"  - {name} (配置: {config})")
        else:
            print(f"  - {name}")
    
    print(f"\n❌ 加载失败的数据集 ({len(failed_datasets)}):")
    for name, config in failed_datasets:
        if config:
            print(f"  - {name} (配置: {config})")
        else:
            print(f"  - {name}")
    
    # 推荐使用的数据集
    if successful_datasets:
        recommended = successful_datasets[0]
        print(f"\n🎯 推荐使用的数据集:")
        if recommended[1]:
            print(f"   数据集名称: {recommended[0]}")
            print(f"   配置: {recommended[1]}")
        else:
            print(f"   数据集名称: {recommended[0]}")
            print(f"   配置: null")

if __name__ == "__main__":
    main()