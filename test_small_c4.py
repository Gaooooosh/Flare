#!/usr/bin/env python3
"""
测试 amanpreet7/allenai-c4 小型数据集
"""

import sys
import os
sys.path.append('/home/xiaoyonggao/Flare')

from utils.simple_dataset_loader import SimpleDatasetLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_small_c4():
    """测试 amanpreet7/allenai-c4 数据集加载"""
    
    # 初始化数据加载器
    cache_dir = "/datacache/huggingface"
    loader = SimpleDatasetLoader(cache_dir=cache_dir)
    
    print("=" * 60)
    print("测试 amanpreet7/allenai-c4 数据集加载")
    print("=" * 60)
    
    try:
        # 尝试加载数据集
        dataset_name = "amanpreet7/allenai-c4"
        print(f"正在加载数据集: {dataset_name}")
        print(f"缓存目录: {cache_dir}")
        
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
                text_preview = first_sample[text_column][:300] + "..." if len(first_sample[text_column]) > 300 else first_sample[text_column]
                print(f"\n文本列 '{text_column}' 预览:")
                print(f"'{text_preview}'")
                print(f"\n文本长度: {len(first_sample[text_column])} 字符")
            else:
                print(f"\n第一个样本内容: {first_sample}")
        
        # 检查数据质量
        print(f"\n📊 数据质量检查:")
        text_lengths = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            if text_column and text_column in sample:
                text_lengths.append(len(sample[text_column]))
        
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            print(f"前10个样本平均文本长度: {avg_length:.1f} 字符")
            print(f"最短文本: {min(text_lengths)} 字符")
            print(f"最长文本: {max(text_lengths)} 字符")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_small_c4()
    if success:
        print("\n🎉 测试成功! amanpreet7/allenai-c4 数据集可以正常加载")
        print("\n💡 建议更新配置文件使用这个数据集")
    else:
        print("\n⚠️  测试失败，将继续使用回退机制")