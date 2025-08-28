#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文档生成器
用于生成详细的训练配置文档和记录
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigDocumentGenerator:
    """配置文档生成器"""
    
    def __init__(self, base_dir: str = "train_config_log"):
        self.template_dir = Path(base_dir)
        self.template_dir.mkdir(exist_ok=True)
    
    def generate_markdown_doc(self, config: Dict[str, Any], output_path: str) -> str:
        """生成Markdown格式的配置文档"""
        
        exp = config.get("experiment", {})
        model = config.get("model", {})
        data = config.get("data", {})
        training = config.get("training", {})
        environment = config.get("environment", {})
        output_config = config.get("output", {})
        
        doc_content = f"""# 训练配置文档

## 实验信息

- **实验名称**: {exp.get('experiment_name', 'N/A')}
- **实验描述**: {exp.get('description', 'N/A')}
- **创建时间**: {exp.get('date', 'N/A')}
- **时间戳**: {exp.get('timestamp', 'N/A')}

---

## 模型配置

### 基础设置
- **模型名称**: `{model.get('model_name', 'N/A')}`
- **最大序列长度**: {model.get('max_seq_length', 'N/A')}
- **数据类型**: {model.get('torch_dtype', 'N/A')}
- **Flash Attention**: {model.get('use_flash_attention', 'N/A')}

### RoPE配置
- **RoPE Theta**: {model.get('rope_theta', 'N/A')}
- **禁用的RoPE层**: {self._format_rope_layers(model.get('no_rope_layers', []))}

{self._generate_rope_analysis(model.get('no_rope_layers', []))}

---

## 数据配置

### 数据集设置
- **数据集名称**: `{data.get('dataset_name', 'N/A')}`
- **数据集配置**: {data.get('dataset_config', 'None')}
- **数据集大小**: {data.get('dataset_size', 'N/A')}
- **验证集比例**: {data.get('validation_split', 'N/A')}

### 数据处理
- **最大长度**: {data.get('max_length', 'N/A')}
- **文本列名**: `{data.get('text_column', 'N/A')}`
- **缓存目录**: `{data.get('cache_dir', 'N/A')}`

---

## 训练配置

### 优化器设置
- **学习率**: {training.get('learning_rate', 'N/A')}
- **批次大小**: {training.get('batch_size', 'N/A')}
- **训练轮数**: {training.get('num_epochs', 'N/A')}
- **预热步数**: {training.get('warmup_steps', 'N/A')}

### 日志和保存
- **日志记录步数**: {training.get('logging_steps', 'N/A')}
- **模型保存步数**: {training.get('save_steps', 'N/A')}
- **评估步数**: {training.get('eval_steps', 'N/A')}

---

## 环境配置

- **GPU IDs**: {environment.get('gpu_ids', 'Auto-detect')}
- **强制CPU**: {environment.get('force_cpu', 'N/A')}

---

## 输出配置

- **基础目录**: `{output_config.get('base_dir', 'N/A')}`
- **实验名称**: `{output_config.get('experiment_name', 'N/A')}`

---

## 配置摘要

{self._generate_config_summary(config)}

---

## 完整配置JSON

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```

---

*文档生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存文档
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return output_path
    
    def _format_rope_layers(self, layers: List[int]) -> str:
        """格式化RoPE层列表"""
        if not layers:
            return "无"
        
        # 将连续的层合并为范围
        ranges = []
        start = layers[0]
        end = layers[0]
        
        for i in range(1, len(layers)):
            if layers[i] == end + 1:
                end = layers[i]
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = layers[i]
        
        # 添加最后一个范围
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)
    
    def _generate_rope_analysis(self, layers: List[int]) -> str:
        """生成RoPE层分析"""
        if not layers:
            return "\n### RoPE层分析\n\n- 所有层都启用RoPE\n- 标准的位置编码配置"
        
        total_layers = 32  # 假设Qwen2.5-3B有32层
        disabled_count = len(layers)
        enabled_count = total_layers - disabled_count
        
        analysis = f"""\n### RoPE层分析\n\n- **总层数**: {total_layers}
- **禁用RoPE层数**: {disabled_count}
- **启用RoPE层数**: {enabled_count}
- **禁用比例**: {disabled_count/total_layers*100:.1f}%

#### 层分布
- **禁用层**: {self._format_rope_layers(layers)}
- **启用层**: {self._format_rope_layers([i for i in range(total_layers) if i not in layers])}

#### 配置说明
这种配置将在指定层禁用RoPE位置编码，可能用于:
- 研究RoPE对不同层的影响
- 优化长序列处理性能
- 实验混合位置编码策略"""
        
        return analysis
    
    def _generate_config_summary(self, config: Dict[str, Any]) -> str:
        """生成配置摘要"""
        model = config.get("model", {})
        data = config.get("data", {})
        training = config.get("training", {})
        
        # 计算预估训练时间和资源
        batch_size = training.get('batch_size', 1)
        num_epochs = training.get('num_epochs', 1)
        dataset_size = data.get('dataset_size', 1000)
        
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * num_epochs
        
        summary = f"""### 关键参数
- **模型**: {model.get('model_name', 'N/A')}
- **上下文长度**: {model.get('max_seq_length', 'N/A')} tokens
- **数据集**: {data.get('dataset_name', 'N/A')} ({dataset_size:,} 样本)
- **训练配置**: {num_epochs} epochs × {batch_size} batch size

### 训练估算
- **每轮步数**: {steps_per_epoch:,}
- **总训练步数**: {total_steps:,}
- **保存检查点**: 每 {training.get('save_steps', 500)} 步
- **评估频率**: 每 {training.get('eval_steps', 500)} 步

### 特殊配置
- **RoPE层禁用**: {len(model.get('no_rope_layers', []))} 层
- **Flash Attention**: {'启用' if model.get('use_flash_attention') else '禁用'}
- **数据类型**: {model.get('torch_dtype', 'N/A')}"""
        
        return summary
    
    def generate_experiment_index(self, log_dir: str = "train_config_log") -> str:
        """生成实验索引文档"""
        log_path = Path(log_dir)
        if not log_path.exists():
            return "No experiments found."
        
        # 收集所有实验配置
        experiments = []
        for config_file in log_path.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                exp_info = config.get("experiment", {})
                model_info = config.get("model", {})
                
                experiments.append({
                    "file": config_file.name,
                    "name": exp_info.get("experiment_name", "Unknown"),
                    "date": exp_info.get("date", "Unknown"),
                    "model": model_info.get("model_name", "Unknown"),
                    "context_length": model_info.get("max_seq_length", "Unknown"),
                    "rope_layers": len(model_info.get("no_rope_layers", [])),
                    "description": exp_info.get("description", "")
                })
            except Exception as e:
                print(f"Warning: Failed to parse {config_file}: {e}")
        
        # 按日期排序
        experiments.sort(key=lambda x: x["date"], reverse=True)
        
        # 生成索引文档
        index_content = f"""# 训练实验索引

*更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 实验列表

| 实验名称 | 日期 | 模型 | 上下文长度 | 禁用RoPE层数 | 描述 | 配置文件 |
|---------|------|------|-----------|-------------|------|----------|
"""
        
        for exp in experiments:
            index_content += f"| {exp['name']} | {exp['date']} | {exp['model']} | {exp['context_length']} | {exp['rope_layers']} | {exp['description']} | [{exp['file']}]({exp['file']}) |\n"
        
        index_content += f"\n\n## 统计信息\n\n- **总实验数**: {len(experiments)}\n"
        
        if experiments:
            models = set(exp['model'] for exp in experiments)
            index_content += f"- **使用的模型**: {', '.join(models)}\n"
            
            context_lengths = set(str(exp['context_length']) for exp in experiments)
            index_content += f"- **上下文长度**: {', '.join(context_lengths)}\n"
        
        # 保存索引
        index_path = log_path / "README.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        return str(index_path)
    
    def create_config_backup(self, config: Dict[str, Any], experiment_name: str, timestamp: str) -> str:
        """创建配置备份"""
        backup_dir = self.template_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"{experiment_name}_{timestamp}_backup.json"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return str(backup_file)
    
    def save_markdown_doc(self, config: Dict[str, Any]) -> str:
        """保存Markdown文档"""
        exp_info = config.get('experiment', {})
        exp_name = exp_info.get('name', exp_info.get('experiment_name', 'unnamed_experiment'))
        timestamp = exp_info.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # 创建文件名
        filename = f"{exp_name}_{timestamp}_config.md"
        filepath = self.template_dir / filename
        
        # 生成并保存文档
        self.generate_markdown_doc(config, str(filepath))
        
        return str(filepath)
    
    def save_json_backup(self, config: Dict[str, Any]) -> str:
        """保存JSON格式的配置备份"""
        exp_info = config.get('experiment', {})
        exp_name = exp_info.get('name', exp_info.get('experiment_name', 'unnamed_experiment'))
        timestamp = exp_info.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        return self.create_config_backup(config, exp_name, timestamp)
    
    def update_experiment_index(self, config: Dict[str, Any]) -> str:
        """更新实验索引文件"""
        return self.generate_experiment_index(str(self.template_dir))


def generate_full_documentation(config: Dict[str, Any], base_dir: str = "train_config_log") -> Dict[str, str]:
    """生成完整的配置文档
    
    Args:
        config: 配置字典
        base_dir: 基础目录
    
    Returns:
        包含各种文档路径的字典
    """
    generator = ConfigDocumentGenerator(base_dir)
    
    results = {}
    
    # 生成Markdown文档
    results['markdown_doc'] = generator.save_markdown_doc(config)
    
    # 生成JSON备份
    results['json_backup'] = generator.save_json_backup(config)
    
    # 更新实验索引
    results['experiment_index'] = generator.update_experiment_index(config)
    
    return results


if __name__ == "__main__":
    # 测试文档生成
    test_config = {
        "experiment": {
            "experiment_name": "test_experiment",
            "description": "测试实验",
            "timestamp": "20240101_120000",
            "date": "2024-01-01 12:00:00"
        },
        "model": {
            "model_name": "Qwen/Qwen2.5-3B",
            "max_seq_length": 4096,
            "rope_theta": 1000000.0,
            "no_rope_layers": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            "use_flash_attention": True,
            "torch_dtype": "bfloat16"
        },
        "data": {
            "dataset_name": "brando/small-c4-dataset",
            "dataset_size": 10000,
            "validation_split": 0.1
        },
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "num_epochs": 3
        }
    }
    
    results = generate_full_documentation(test_config)
    print("Generated documentation:")
    for doc_type, path in results.items():
        print(f"  {doc_type}: {path}")