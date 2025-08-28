#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的配置管理器
统一管理训练配置，避免复杂的参数传递
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, fields

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-3B"
    rope_theta: float = 1000000.0
    max_seq_length: int = 4096
    no_rope_layers: list = field(default_factory=lambda: list(range(20, 33)))
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "togethercomputer/RedPajama-Data-1T-Sample"
    dataset_size: Optional[int] = 10000
    validation_split: float = 0.1
    max_length: int = 4096
    text_column: str = "text"
    cache_dir: Optional[str] = None


@dataclass
class TrainingConfig:
    """训练配置"""
    stage: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "./output"
    experiment_name: Optional[str] = None


@dataclass
class EnvironmentConfig:
    """环境配置"""
    gpu_ids: Optional[list] = None
    force_cpu: bool = False


@dataclass
class SimpleConfig:
    """简化的配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model': {
                'model_name': self.model.model_name,
                'rope_theta': self.model.rope_theta,
                'max_seq_length': self.model.max_seq_length,
                'no_rope_layers': self.model.no_rope_layers,
                'use_flash_attention': self.model.use_flash_attention,
                'torch_dtype': self.model.torch_dtype
            },
            'data': {
                'dataset_name': self.data.dataset_name,
                'dataset_size': self.data.dataset_size,
                'validation_split': self.data.validation_split
            },
            'training': {
                'stage': self.training.stage,
                'learning_rate': self.training.learning_rate,
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'warmup_steps': self.training.warmup_steps,
                'logging_steps': self.training.logging_steps,
                'save_steps': self.training.save_steps,
                'eval_steps': self.training.eval_steps
            },
            'output': {
                'base_dir': self.output.base_dir,
                'experiment_name': self.output.experiment_name
            },
            'environment': {
                'gpu_ids': self.environment.gpu_ids,
                'force_cpu': self.environment.force_cpu
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleConfig':
        """从字典创建配置"""
        config = cls()
        
        # 处理模型配置
        if 'model' in data:
            model_data = data['model']
            config.model = ModelConfig(
                model_name=model_data.get('model_name', config.model.model_name),
                rope_theta=model_data.get('rope_theta', config.model.rope_theta),
                max_seq_length=model_data.get('max_seq_length', config.model.max_seq_length),
                no_rope_layers=model_data.get('no_rope_layers', config.model.no_rope_layers),
                use_flash_attention=model_data.get('use_flash_attention', config.model.use_flash_attention),
                torch_dtype=model_data.get('torch_dtype', config.model.torch_dtype)
            )
        
        # 处理数据配置
        if 'data' in data:
            data_config = data['data']
            config.data = DataConfig(
                dataset_name=data_config.get('dataset_name', config.data.dataset_name),
                dataset_size=data_config.get('dataset_size', config.data.dataset_size),
                validation_split=data_config.get('validation_split', config.data.validation_split)
            )
        
        # 处理训练配置
        if 'training' in data:
            training_data = data['training']
            config.training = TrainingConfig(
                stage=training_data.get('stage', config.training.stage),
                learning_rate=training_data.get('learning_rate', config.training.learning_rate),
                batch_size=training_data.get('batch_size', config.training.batch_size),
                num_epochs=training_data.get('num_epochs', config.training.num_epochs),
                warmup_steps=training_data.get('warmup_steps', config.training.warmup_steps),
                logging_steps=training_data.get('logging_steps', config.training.logging_steps),
                save_steps=training_data.get('save_steps', config.training.save_steps),
                eval_steps=training_data.get('eval_steps', config.training.eval_steps)
            )
        
        # 处理输出配置
        if 'output' in data:
            output_data = data['output']
            config.output = OutputConfig(
                base_dir=output_data.get('base_dir', config.output.base_dir),
                experiment_name=output_data.get('experiment_name', config.output.experiment_name)
            )
        
        # 处理环境配置
        if 'environment' in data:
            env_data = data['environment']
            config.environment = EnvironmentConfig(
                gpu_ids=env_data.get('gpu_ids', config.environment.gpu_ids),
                force_cpu=env_data.get('force_cpu', config.environment.force_cpu)
            )
        
        return config


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = SimpleConfig()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """加载配置文件"""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 简化配置文件结构，扁平化处理
            flattened = self._flatten_config(data)
            self.config = SimpleConfig.from_dict(flattened)
            logger.info(f"成功加载配置文件: {config_path}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
    
    def _flatten_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """扁平化配置文件结构，兼容旧格式"""
        # 如果已经是新格式，直接返回
        if all(key in data for key in ['model', 'data', 'training', 'output', 'environment']):
            return data
        
        # 兼容旧的扁平化格式
        result = {
            'model': {},
            'data': {},
            'training': {},
            'output': {},
            'environment': {}
        }
        
        # 映射旧格式到新格式
        for key, value in data.items():
            if key in ['model_name', 'model_name_or_path']:
                result['model']['model_name'] = value
            elif key in ['rope_theta', 'max_seq_length', 'no_rope_layers', 'use_flash_attention', 'torch_dtype']:
                result['model'][key] = value
            elif key in ['dataset_name', 'dataset_size', 'validation_split']:
                result['data'][key] = value
            elif key in ['stage', 'learning_rate', 'num_epochs', 'warmup_steps', 'logging_steps', 'save_steps', 'eval_steps']:
                result['training'][key] = value
            elif key == 'per_device_train_batch_size':
                result['training']['batch_size'] = value
            elif key == 'batch_size':
                result['training']['batch_size'] = value
            elif key in ['base_dir', 'output_dir']:
                result['output']['base_dir'] = value
            elif key == 'experiment_name':
                result['output']['experiment_name'] = value
            elif key in ['gpu_ids', 'force_cpu']:
                result['environment'][key] = value
        
        return result
    
    def save_config(self, output_path: str) -> None:
        """保存配置文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            # 支持嵌套配置更新
            if '.' in key:
                section, field = key.split('.', 1)
                if hasattr(self.config, section):
                    section_obj = getattr(self.config, section)
                    if hasattr(section_obj, field):
                        setattr(section_obj, field, value)
                        logger.info(f"配置已更新: {key} = {value}")
                    else:
                        logger.warning(f"未知配置项: {key}")
                else:
                    logger.warning(f"未知配置节: {section}")
            else:
                # 兼容旧的扁平化配置更新
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"配置已更新: {key} = {value}")
                else:
                    logger.warning(f"未知配置项: {key}")
    
    def get_config(self) -> SimpleConfig:
        """获取配置对象"""
        return self.config
    
    def print_config(self) -> None:
        """打印配置信息"""
        print("\n" + "=" * 50)
        print("当前训练配置")
        print("=" * 50)
        
        print("\n[模型配置]")
        print(f"  模型名称: {self.config.model.model_name}")
        print(f"  最大序列长度: {self.config.model.max_seq_length}")
        print(f"  RoPE theta: {self.config.model.rope_theta}")
        print(f"  使用Flash Attention: {self.config.model.use_flash_attention}")
        print(f"  数据类型: {self.config.model.torch_dtype}")
        
        print("\n[数据配置]")
        print(f"  数据集名称: {self.config.data.dataset_name}")
        print(f"  数据集大小: {self.config.data.dataset_size}")
        print(f"  验证集比例: {self.config.data.validation_split}")
        
        print("\n[训练配置]")
        print(f"  训练阶段: {self.config.training.stage}")
        print(f"  学习率: {self.config.training.learning_rate}")
        print(f"  批次大小: {self.config.training.batch_size}")
        print(f"  训练轮数: {self.config.training.num_epochs}")
        print(f"  预热步数: {self.config.training.warmup_steps}")
        
        print("\n[输出配置]")
        print(f"  输出目录: {self.config.output.base_dir}")
        print(f"  实验名称: {self.config.output.experiment_name}")
        
        print("\n[环境配置]")
        print(f"  GPU设备: {self.config.environment.gpu_ids}")
        print(f"  强制CPU: {self.config.environment.force_cpu}")
        
        print("=" * 50 + "\n")