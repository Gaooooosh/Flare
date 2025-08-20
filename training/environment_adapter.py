#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境适配器模块

自动检测运行环境并调整训练参数以适应不同的硬件配置
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentInfo:
    """环境信息"""
    use_cpu: bool
    gpu_count: int
    gpu_memory_gb: float
    cpu_cores: int
    system_memory_gb: float
    cuda_available: bool
    mps_available: bool  # Apple Metal Performance Shaders
    
    def __str__(self) -> str:
        return f"EnvironmentInfo(use_cpu={self.use_cpu}, gpu_count={self.gpu_count}, gpu_memory_gb={self.gpu_memory_gb:.1f}, cpu_cores={self.cpu_cores}, system_memory_gb={self.system_memory_gb:.1f})"

class EnvironmentAdapter:
    """环境适配器"""
    
    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self.env_info = self._detect_environment()
        logger.info(f"检测到环境: {self.env_info}")
    
    def _detect_environment(self) -> EnvironmentInfo:
        """检测运行环境"""
        # 检测CUDA可用性
        cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # 检测MPS可用性（Apple Silicon）
        mps_available = TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # GPU信息
        gpu_count = 0
        gpu_memory_gb = 0.0
        
        if cuda_available and not self.force_cpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # 获取第一个GPU的内存信息
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / 1024 / 1024 / 1024
        
        # CPU信息
        cpu_cores = os.cpu_count() or 1
        
        # 系统内存信息
        system_memory_gb = 8.0  # 默认值
        if PSUTIL_AVAILABLE:
            system_memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        # 决定是否使用CPU
        use_cpu = self.force_cpu or not cuda_available or gpu_count == 0
        
        return EnvironmentInfo(
            use_cpu=use_cpu,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory_gb,
            cuda_available=cuda_available,
            mps_available=mps_available
        )
    
    def adapt_training_args(self, training_args) -> Any:
        """根据环境调整训练参数"""
        if self.env_info.use_cpu:
            # CPU模式调整
            logger.info("调整训练参数以适应CPU环境")
            
            # 禁用GPU相关设置
            training_args.no_cuda = True
            training_args.use_cpu = True
            
            # 调整数据类型
            training_args.fp16 = False
            training_args.bf16 = False
            
            # 调整批次大小（如果太大）
            if hasattr(training_args, 'per_device_train_batch_size'):
                if training_args.per_device_train_batch_size > 4:
                    original_batch_size = training_args.per_device_train_batch_size
                    training_args.per_device_train_batch_size = min(4, original_batch_size)
                    # 增加梯度累积步数以保持有效批次大小
                    if hasattr(training_args, 'gradient_accumulation_steps'):
                        training_args.gradient_accumulation_steps = max(
                            training_args.gradient_accumulation_steps,
                            original_batch_size // training_args.per_device_train_batch_size
                        )
                    logger.info(f"调整批次大小: {original_batch_size} -> {training_args.per_device_train_batch_size}")
            
            if hasattr(training_args, 'per_device_eval_batch_size'):
                training_args.per_device_eval_batch_size = min(4, training_args.per_device_eval_batch_size)
            
            # 调整数据加载器设置
            if hasattr(training_args, 'dataloader_num_workers'):
                training_args.dataloader_num_workers = min(2, self.env_info.cpu_cores // 2)
            
            if hasattr(training_args, 'dataloader_pin_memory'):
                training_args.dataloader_pin_memory = False
            
            # 启用梯度检查点以节省内存
            if hasattr(training_args, 'gradient_checkpointing'):
                training_args.gradient_checkpointing = True
            
        else:
            # GPU模式调整
            logger.info("调整训练参数以适应GPU环境")
            
            # 启用GPU设置
            training_args.no_cuda = False
            
            # 根据GPU内存调整设置
            if self.env_info.gpu_memory_gb < 8:
                # 小显存GPU
                logger.info("检测到小显存GPU，启用内存优化")
                if hasattr(training_args, 'gradient_checkpointing'):
                    training_args.gradient_checkpointing = True
                if hasattr(training_args, 'fp16') and not training_args.bf16:
                    training_args.fp16 = True
            
            # 多GPU设置
            if self.env_info.gpu_count > 1:
                logger.info(f"检测到{self.env_info.gpu_count}个GPU，启用多GPU训练")
                # 这里可以添加多GPU相关的设置
        
        return training_args
    
    def adapt_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据环境调整模型配置"""
        adapted_config = model_config.copy()
        
        if self.env_info.use_cpu:
            # CPU模式调整
            adapted_config['torch_dtype'] = 'float32'
            adapted_config['device_map'] = None
            
            # 禁用Flash Attention（CPU不支持）
            if 'use_flash_attention_2' in adapted_config:
                adapted_config['use_flash_attention_2'] = False
            
            # 调整注意力实现
            if 'attn_implementation' in adapted_config:
                adapted_config['attn_implementation'] = 'eager'
        
        else:
            # GPU模式调整
            if self.env_info.gpu_memory_gb < 8:
                # 小显存优化
                adapted_config['torch_dtype'] = 'float16'
            elif self.env_info.gpu_memory_gb >= 16:
                # 大显存可以使用更高精度
                adapted_config['torch_dtype'] = 'bfloat16'
            
            # 多GPU设置
            if self.env_info.gpu_count > 1:
                adapted_config['device_map'] = 'auto'
        
        return adapted_config
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """获取推荐设置"""
        settings = {
            'environment': str(self.env_info),
            'recommendations': []
        }
        
        if self.env_info.use_cpu:
            settings['recommendations'].extend([
                "使用较小的批次大小（1-4）",
                "启用梯度检查点以节省内存",
                "使用float32精度",
                "禁用Flash Attention",
                "减少数据加载器工作进程数"
            ])
        else:
            if self.env_info.gpu_memory_gb < 8:
                settings['recommendations'].extend([
                    "启用混合精度训练（fp16）",
                    "启用梯度检查点",
                    "使用较小的批次大小"
                ])
            elif self.env_info.gpu_memory_gb >= 16:
                settings['recommendations'].extend([
                    "可以使用较大的批次大小",
                    "考虑使用bfloat16精度",
                    "可以启用Flash Attention 2"
                ])
            
            if self.env_info.gpu_count > 1:
                settings['recommendations'].append(f"利用{self.env_info.gpu_count}个GPU进行并行训练")
        
        return settings
    
    def validate_environment(self) -> bool:
        """验证环境是否适合训练"""
        issues = []
        
        # 检查内存
        if self.env_info.system_memory_gb < 4:
            issues.append("系统内存不足4GB，可能影响训练性能")
        
        if not self.env_info.use_cpu and self.env_info.gpu_memory_gb < 4:
            issues.append("GPU内存不足4GB，建议使用CPU模式或减少模型大小")
        
        # 检查CPU核心数
        if self.env_info.cpu_cores < 2:
            issues.append("CPU核心数过少，可能影响数据加载性能")
        
        # 输出问题
        if issues:
            logger.warning("环境验证发现以下问题:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("环境验证通过")
            return True