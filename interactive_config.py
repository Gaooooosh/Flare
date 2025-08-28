#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式训练配置脚本
允许用户通过交互方式配置训练参数
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.config_manager import ConfigManager, SimpleConfig
from utils.config_doc_generator import generate_full_documentation


class InteractiveConfigGenerator:
    """交互式配置生成器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.experiment_name = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_user_input(self, prompt: str, default: Any = None, input_type: type = str) -> Any:
        """获取用户输入"""
        if default is not None:
            full_prompt = f"{prompt} (默认: {default}): "
        else:
            full_prompt = f"{prompt}: "
            
        while True:
            try:
                user_input = input(full_prompt).strip()
                
                # 如果用户没有输入且有默认值，使用默认值
                if not user_input and default is not None:
                    return default
                    
                # 如果没有输入且没有默认值，继续询问
                if not user_input:
                    print("请输入有效值！")
                    continue
                    
                # 类型转换
                if input_type == int:
                    return int(user_input)
                elif input_type == float:
                    return float(user_input)
                elif input_type == bool:
                    return user_input.lower() in ['true', 'yes', 'y', '1', 'on']
                elif input_type == list:
                    # 解析列表输入，支持逗号分隔或范围
                    return self._parse_list_input(user_input)
                else:
                    return user_input
                    
            except ValueError as e:
                print(f"输入格式错误: {e}，请重新输入！")
            except KeyboardInterrupt:
                print("\n配置已取消")
                sys.exit(0)
                
    def _parse_list_input(self, input_str: str) -> List[int]:
        """解析列表输入，支持逗号分隔和范围"""
        result = []
        parts = input_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part and not part.startswith('-'):
                # 范围输入，如 "20-32"
                start, end = map(int, part.split('-'))
                result.extend(range(start, end + 1))
            else:
                # 单个数字
                result.append(int(part))
                
        return sorted(list(set(result)))  # 去重并排序
    
    def collect_experiment_info(self) -> Dict[str, str]:
        """收集实验信息"""
        print("\n=== 实验信息配置 ===")
        
        self.experiment_name = self.get_user_input(
            "实验名称", 
            f"rope_experiment_{self.timestamp}"
        )
        
        description = self.get_user_input(
            "实验描述", 
            "RoPE层禁用实验"
        )
        
        return {
            "experiment_name": self.experiment_name,
            "description": description,
            "timestamp": self.timestamp,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def collect_model_config(self) -> Dict[str, Any]:
        """收集模型配置"""
        print("\n=== 模型配置 ===")
        
        model_name = self.get_user_input(
            "模型名称", 
            "Qwen/Qwen2.5-3B"
        )
        
        max_seq_length = self.get_user_input(
            "训练上下文长度 (max_seq_length)", 
            4096, 
            int
        )
        
        rope_theta = self.get_user_input(
            "RoPE theta 值", 
            1000000.0, 
            float
        )
        
        print("\n禁用的RoPE层配置:")
        print("输入格式示例:")
        print("  - 单个层: 20,21,22")
        print("  - 范围: 20-32")
        print("  - 混合: 20-25,30,31")
        
        no_rope_layers = self.get_user_input(
            "禁用的RoPE层", 
            "20-32", 
            list
        )
        
        use_flash_attention = self.get_user_input(
            "使用Flash Attention", 
            True, 
            bool
        )
        
        torch_dtype = self.get_user_input(
            "PyTorch数据类型", 
            "bfloat16"
        )
        
        return {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "rope_theta": rope_theta,
            "no_rope_layers": no_rope_layers,
            "use_flash_attention": use_flash_attention,
            "torch_dtype": torch_dtype
        }
    
    def collect_data_config(self) -> Dict[str, Any]:
        """收集数据配置"""
        print("\n=== 数据配置 ===")
        
        dataset_name = self.get_user_input(
            "数据集名称", 
            "brando/small-c4-dataset"
        )
        
        dataset_size = self.get_user_input(
            "数据集大小 (None表示使用全部)", 
            10000, 
            int
        )
        
        validation_split = self.get_user_input(
            "验证集比例", 
            0.1, 
            float
        )
        
        max_length = self.get_user_input(
            "最大序列长度", 
            4096, 
            int
        )
        
        text_column = self.get_user_input(
            "文本列名", 
            "text"
        )
        
        cache_dir = self.get_user_input(
            "缓存目录", 
            "/datacache/huggingface"
        )
        
        return {
            "dataset_name": dataset_name,
            "dataset_config": None,
            "dataset_size": dataset_size,
            "validation_split": validation_split,
            "max_length": max_length,
            "text_column": text_column,
            "cache_dir": cache_dir
        }
    
    def collect_training_config(self) -> Dict[str, Any]:
        """收集训练配置"""
        print("\n=== 训练配置 ===")
        
        learning_rate = self.get_user_input(
            "学习率", 
            1e-4, 
            float
        )
        
        batch_size = self.get_user_input(
            "批次大小", 
            2, 
            int
        )
        
        num_epochs = self.get_user_input(
            "训练轮数", 
            3, 
            int
        )
        
        warmup_steps = self.get_user_input(
            "预热步数", 
            100, 
            int
        )
        
        logging_steps = self.get_user_input(
            "日志记录步数", 
            10, 
            int
        )
        
        save_steps = self.get_user_input(
            "模型保存步数", 
            500, 
            int
        )
        
        eval_steps = self.get_user_input(
            "评估步数", 
            500, 
            int
        )
        
        return {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps
        }
    
    def collect_environment_config(self) -> Dict[str, Any]:
        """收集环境配置"""
        print("\n=== 环境配置 ===")
        
        gpu_ids_input = self.get_user_input(
            "GPU IDs (逗号分隔，留空自动检测)", 
            ""
        )
        
        gpu_ids = None
        if gpu_ids_input:
            gpu_ids = [int(x.strip()) for x in gpu_ids_input.split(',')]
        
        force_cpu = self.get_user_input(
            "强制使用CPU", 
            False, 
            bool
        )
        
        return {
            "gpu_ids": gpu_ids,
            "force_cpu": force_cpu
        }
    
    def generate_config(self) -> Dict[str, Any]:
        """生成完整配置"""
        print("\n" + "=" * 60)
        print("欢迎使用交互式训练配置生成器")
        print("=" * 60)
        
        # 收集各部分配置
        experiment_info = self.collect_experiment_info()
        model_config = self.collect_model_config()
        data_config = self.collect_data_config()
        training_config = self.collect_training_config()
        environment_config = self.collect_environment_config()
        
        # 组装完整配置
        full_config = {
            "experiment": experiment_info,
            "model": model_config,
            "data": data_config,
            "training": training_config,
            "environment": environment_config,
            "output": {
                "base_dir": "./output",
                "experiment_name": self.experiment_name
            }
        }
        
        return full_config
    
    def save_config_log(self, config: Dict[str, Any]) -> str:
        """保存配置日志"""
        # 创建配置日志目录
        log_dir = Path("train_config_log")
        log_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        filename = f"{self.experiment_name}_{self.timestamp}.json"
        log_path = log_dir / filename
        
        # 保存配置
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return str(log_path)
    
    def save_simple_config(self, config: Dict[str, Any]) -> str:
        """保存简化配置文件供训练使用"""
        simple_config = {
            "model": config["model"],
            "data": config["data"],
            "training": config["training"],
            "environment": config["environment"],
            "output": config["output"]
        }
        
        config_path = "simple_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(simple_config, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def print_config_summary(self, config: Dict[str, Any]):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("配置摘要")
        print("=" * 60)
        
        exp = config["experiment"]
        model = config["model"]
        data = config["data"]
        training = config["training"]
        
        print(f"实验名称: {exp['experiment_name']}")
        print(f"实验描述: {exp['description']}")
        print(f"时间戳: {exp['date']}")
        print()
        print(f"模型: {model['model_name']}")
        print(f"上下文长度: {model['max_seq_length']}")
        print(f"禁用RoPE层: {model['no_rope_layers']}")
        print(f"RoPE theta: {model['rope_theta']}")
        print()
        print(f"数据集: {data['dataset_name']}")
        print(f"数据集大小: {data['dataset_size']}")
        print(f"验证集比例: {data['validation_split']}")
        print()
        print(f"学习率: {training['learning_rate']}")
        print(f"批次大小: {training['batch_size']}")
        print(f"训练轮数: {training['num_epochs']}")
        print("=" * 60)


def main():
    """主函数"""
    try:
        generator = InteractiveConfigGenerator()
        
        # 生成配置
        config = generator.generate_config()
        
        # 打印摘要
        generator.print_config_summary(config)
        
        # 确认保存
        confirm = input("\n确认保存配置? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            print("\n正在生成配置文档...")
            
            # 生成完整文档
            doc_results = generate_full_documentation(config)
            
            print("\n✅ 配置文档生成完成:")
            for doc_type, path in doc_results.items():
                print(f"  📄 {doc_type}: {path}")
            
            # 保存简化配置
            config_path = generator.save_simple_config(config)
            print(f"\n✅ 训练配置已保存: {config_path}")
            
            print(f"\n📁 配置记录目录: train_config_log/")
            print(f"📋 实验索引: train_config_log/README.md")
            
            print(f"\n🚀 现在可以使用以下命令开始训练:")
            print(f"   python training/train_simple.py")
            print(f"\n📖 查看完整配置文档:")
            print(f"   cat {doc_results.get('markdown_doc', 'N/A')}")
        else:
            print("\n配置已取消")
            
    except KeyboardInterrupt:
        print("\n\n配置已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 配置生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()