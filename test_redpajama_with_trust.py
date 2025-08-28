#!/usr/bin/env python3
"""
测试添加 trust_remote_code=True 后的 RedPajama 数据集加载
"""

import sys
import os
sys.path.append('/home/xiaoyonggao/Flare')

from utils.simple_dataset_loader import SimpleDatasetLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_redpajama_loading():
    """测试 RedPajama-Data-1T-Sample 数据集加载"""
    
    # 初始化数据加载器
    cache_dir = "/datacache/huggingface"
    loader = SimpleDatasetLoader(cache_dir=cache_dir)
    
    print("=" * 60)
    print("测试 RedPajama-Data-1T-Sample 数据集加载")
    print("=" * 60)
    
    try:
        # 尝试加载数据集
        dataset_name = "togethercomputer/RedPajama-Data-1T-Sample"
        print(f"正在加载数据集: {dataset_name}")
        print(f"缓存目录: {cache_dir}")
        print(f"使用 trust_remote_code=True")
        
        dataset = loader.load_dataset(
            dataset_name=dataset_name,
            dataset_config=None,
            split="train",
            size_limit=1000  # 限制为1000个样本进行测试
        )
        
        print(f"\n✅ 数据集加载成功!")
        print(f"数据集大小: {len(dataset)}")
        print(f"数据集列名: {dataset.column_names}")
        
        # 查看第一个样本
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"\n第一个样本的键: {list(first_sample.keys())}")
            
            # 查找文本列
            text_column = None
            for col in ['text', 'content', 'document', 'raw_content']:
                if col in first_sample:
                    text_column = col
                    break
            
            if text_column:
                text_preview = first_sample[text_column][:200] + "..." if len(first_sample[text_column]) > 200 else first_sample[text_column]
                print(f"\n文本列 '{text_column}' 预览:")
                print(f"'{text_preview}'")
            else:
                print(f"\n第一个样本内容: {first_sample}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_redpajama_loading()
    if success:
        print("\n🎉 测试成功! RedPajama 数据集可以正常加载")
    else:
        print("\n⚠️  测试失败，将使用回退机制")