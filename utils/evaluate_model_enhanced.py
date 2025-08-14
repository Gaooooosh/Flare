#!/usr/bin/env python3
"""
增强版模型评估脚本
支持功能：
1. 困惑度(PPL)计算
2. 多种评估数据集
3. 长上下文评估
4. 详细的评估报告
5. TensorBoard记录
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Hugging Face imports
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from patch_qwen_rope import patch_qwen_rope

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 device: str = "auto",
                 torch_dtype: str = "bfloat16",
                 no_rope_layers: Optional[List[int]] = None,
                 rope_theta: float = 1000000.0,
                 use_flash_attention: bool = True):
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device
        self.no_rope_layers = no_rope_layers or []
        self.rope_theta = rope_theta
        
        # 数据类型映射
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_mapping.get(torch_dtype, torch.bfloat16)
        
        # 加载模型和分词器
        self._load_model_and_tokenizer(use_flash_attention)
    
    def _load_model_and_tokenizer(self, use_flash_attention: bool):
        """加载模型和分词器"""
        logger.info(f"加载分词器: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"加载模型: {self.model_path}")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"":self.device}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        # 应用RoPE修改
        if self.no_rope_layers:
            patch_qwen_rope(self.model, no_rope_layers=self.no_rope_layers)
            logger.info(f"已禁用层 {self.no_rope_layers} 的RoPE")
        
        # 设置RoPE参数
        if hasattr(self.model.config, "rope_theta"):
            self.model.config.rope_theta = self.rope_theta
            logger.info(f"设置RoPE theta为: {self.rope_theta}")
        
        self.model.eval()
        logger.info("模型加载完成")
    
    def compute_perplexity(self, 
                          dataset: Dataset,
                          max_length: int = 2048,
                          batch_size: int = 1,
                          max_samples: Optional[int] = None) -> Dict[str, float]:
        """计算困惑度"""
        logger.info("开始计算困惑度...")
        
        # 限制样本数量
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # 数据预处理
        def tokenize_function(examples):
            # 尝试不同的文本列名
            text_key = None
            for key in ["text", "content", "document", "raw_content"]:
                if key in examples:
                    text_key = key
                    break
            
            if text_key is None:
                raise ValueError(f"找不到文本列，可用列: {list(examples.keys())}")
            
            texts = examples[text_key]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_overflowing_tokens=False,
            )
            
            return tokenized
        
        # 分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="分词数据"
        )
        
        # 过滤空样本
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        
        # 创建数据加载器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算困惑度"):
                # 移动到设备
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                
                # 计算有效token数量
                valid_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
                num_batches += 1
        
        # 计算平均损失和困惑度
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        
        results = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_samples": len(tokenized_dataset),
            "num_batches": num_batches
        }
        
        logger.info(f"困惑度计算完成: PPL={perplexity:.4f}, Loss={avg_loss:.4f}")
        return results
    
    def evaluate_on_multiple_datasets(self,
                                    dataset_configs: List[Dict[str, Any]],
                                    output_dir: str,
                                    max_length: int = 2048,
                                    batch_size: int = 1,
                                    max_samples_per_dataset: int = 1000) -> Dict[str, Dict[str, float]]:
        """在多个数据集上评估"""
        results = {}
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建TensorBoard writer
        tb_writer = SummaryWriter(log_dir=str(output_path / "tensorboard"))
        
        for i, config in enumerate(dataset_configs):
            dataset_name = config.get("name", f"dataset_{i}")
            logger.info(f"评估数据集: {dataset_name}")
            
            try:
                # 加载数据集
                if "path" in config:
                    # Hugging Face数据集
                    load_kwargs = {
                        "path": config["path"],
                        "split": config.get("split", "test")
                    }
                    if "name" in config:
                        load_kwargs["name"] = config["name"]
                    
                    dataset = load_dataset(**load_kwargs)
                elif "data" in config:
                    # 自定义数据
                    dataset = Dataset.from_dict(config["data"])
                else:
                    logger.warning(f"跳过数据集 {dataset_name}: 缺少数据源配置")
                    continue
                
                # 计算困惑度
                ppl_results = self.compute_perplexity(
                    dataset=dataset,
                    max_length=max_length,
                    batch_size=batch_size,
                    max_samples=max_samples_per_dataset
                )
                
                results[dataset_name] = ppl_results
                
                # 记录到TensorBoard
                tb_writer.add_scalar(f"perplexity/{dataset_name}", ppl_results["perplexity"], 0)
                tb_writer.add_scalar(f"loss/{dataset_name}", ppl_results["avg_loss"], 0)
                
                logger.info(f"数据集 {dataset_name} 评估完成: PPL={ppl_results['perplexity']:.4f}")
                
            except Exception as e:
                logger.error(f"评估数据集 {dataset_name} 时出错: {e}")
                results[dataset_name] = {"error": str(e)}
        
        tb_writer.close()
        
        # 保存结果
        results_file = output_path / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果保存在: {results_file}")
        return results
    
    def generate_evaluation_report(self, 
                                 results: Dict[str, Dict[str, float]],
                                 output_dir: str):
        """生成评估报告"""
        output_path = Path(output_dir)
        
        # 生成Markdown报告
        report_lines = [
            "# 模型评估报告\n",
            f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**模型路径**: {self.model_path}\n",
            f"**RoPE禁用层**: {self.no_rope_layers}\n",
            f"**RoPE Theta**: {self.rope_theta}\n\n",
            "## 评估结果\n\n",
            "| 数据集 | 困惑度 | 平均损失 | 样本数 | Token数 |\n",
            "|--------|--------|----------|--------|---------|\n"
        ]
        
        for dataset_name, result in results.items():
            if "error" in result:
                report_lines.append(f"| {dataset_name} | ERROR | - | - | - |\n")
            else:
                report_lines.append(
                    f"| {dataset_name} | {result['perplexity']:.4f} | "
                    f"{result['avg_loss']:.4f} | {result['num_samples']} | "
                    f"{result['total_tokens']} |\n"
                )
        
        # 添加统计信息
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            ppls = [r["perplexity"] for r in valid_results.values()]
            report_lines.extend([
                "\n## 统计信息\n\n",
                f"**平均困惑度**: {np.mean(ppls):.4f}\n",
                f"**困惑度标准差**: {np.std(ppls):.4f}\n",
                f"**最低困惑度**: {np.min(ppls):.4f}\n",
                f"**最高困惑度**: {np.max(ppls):.4f}\n"
            ])
        
        # 保存报告
        report_file = output_path / "evaluation_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        
        logger.info(f"评估报告保存在: {report_file}")


def get_default_dataset_configs() -> List[Dict[str, Any]]:
    """获取默认评估数据集配置"""
    return [
        {
            "name": "wikitext-2",
            "path": "wikitext",
            "name": "wikitext-2-raw-v1",
            "split": "test"
        },
        {
            "name": "wikitext-103",
            "path": "wikitext",
            "name": "wikitext-103-raw-v1",
            "split": "test"
        },
        {
            "name": "ptb",
            "path": "ptb_text_only",
            "split": "test"
        },
        {
            "name": "lambada",
            "path": "lambada",
            "split": "test"
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="分词器路径（默认与模型路径相同）")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备：auto, cuda:0, cpu等")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="模型数据类型")
    parser.add_argument("--no_rope_layers", type=int, nargs="*", 
                       default=list(range(20, 33)),
                       help="禁用RoPE的层号列表")
    parser.add_argument("--rope_theta", type=float, default=1000000.0,
                       help="RoPE基频")
    parser.add_argument("--use_flash_attention", action="store_true",
                       help="使用Flash Attention 2")
    
    # 评估参数
    parser.add_argument("--output_dir", type=str, 
                       default="/work/xiaoyonggao/evaluation_results",
                       help="输出目录")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--max_samples_per_dataset", type=int, default=1000,
                       help="每个数据集的最大样本数")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="自定义数据集配置文件路径（JSON格式）")
    parser.add_argument("--use_default_datasets", action="store_true",
                       help="使用默认评估数据集")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        no_rope_layers=args.no_rope_layers,
        rope_theta=args.rope_theta,
        use_flash_attention=args.use_flash_attention
    )
    
    # 准备数据集配置
    if args.dataset_config:
        # 从文件加载配置
        with open(args.dataset_config, "r", encoding="utf-8") as f:
            dataset_configs = json.load(f)
    elif args.use_default_datasets:
        # 使用默认数据集
        dataset_configs = get_default_dataset_configs()
    else:
        # 使用示例数据集
        dataset_configs = [
            {
                "name": "sample_data",
                "data": {
                    "text": [
                        "这是一个测试文本，用于评估模型的困惑度。" * 10,
                        "人工智能是计算机科学的一个分支，它企图了解智能的实质。" * 10,
                        "深度学习是机器学习的一个子领域，基于人工神经网络。" * 10
                    ] * 100
                }
            }
        ]
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"evaluation_{timestamp}"
    
    # 执行评估
    logger.info("开始模型评估...")
    results = evaluator.evaluate_on_multiple_datasets(
        dataset_configs=dataset_configs,
        output_dir=str(output_dir),
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples_per_dataset=args.max_samples_per_dataset
    )
    
    # 生成报告
    evaluator.generate_evaluation_report(results, str(output_dir))
    
    # 打印结果摘要
    logger.info("\n=== 评估结果摘要 ===")
    for dataset_name, result in results.items():
        if "error" in result:
            logger.info(f"{dataset_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{dataset_name}: PPL={result['perplexity']:.4f}, Loss={result['avg_loss']:.4f}")
    
    logger.info(f"\n详细结果保存在: {output_dir}")


if __name__ == "__main__":
    main()