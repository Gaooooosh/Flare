#!/usr/bin/env python3
"""
环境适配模块
提供CPU/GPU环境检测和适配功能，确保训练代码在不同环境下都能正常运行
"""

import os
import logging
import torch
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """环境信息"""
    has_cuda: bool
    cuda_device_count: int
    cuda_version: Optional[str]
    pytorch_version: str
    available_memory_gb: float
    cpu_count: int
    use_cpu: bool
    recommended_batch_size: int
    recommended_workers: int
    recommended_precision: str


class EnvironmentAdapter:
    """
    环境适配器
    自动检测硬件环境并提供相应的优化建议
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        初始化环境适配器
        
        Args:
            force_cpu: 是否强制使用CPU模式
        """
        self.force_cpu = force_cpu
        self.env_info = self._detect_environment()
        self._setup_environment()
    
    def _detect_environment(self) -> EnvironmentInfo:
        """
        检测当前环境信息
        
        Returns:
            EnvironmentInfo: 环境信息
        """
        logger.info("检测环境信息...")
        
        # 检测CUDA
        has_cuda = torch.cuda.is_available() and not self.force_cpu
        cuda_device_count = torch.cuda.device_count() if has_cuda else 0
        cuda_version = torch.version.cuda if has_cuda else None
        
        # 检测内存
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        
        # CPU信息
        cpu_count = os.cpu_count() or 1
        
        # 确定是否使用CPU
        use_cpu = not has_cuda or self.force_cpu
        
        # 推荐配置
        recommended_batch_size = self._recommend_batch_size(has_cuda, available_memory_gb)
        recommended_workers = self._recommend_workers(use_cpu, cpu_count)
        recommended_precision = self._recommend_precision(has_cuda)
        
        env_info = EnvironmentInfo(
            has_cuda=has_cuda,
            cuda_device_count=cuda_device_count,
            cuda_version=cuda_version,
            pytorch_version=torch.__version__,
            available_memory_gb=available_memory_gb,
            cpu_count=cpu_count,
            use_cpu=use_cpu,
            recommended_batch_size=recommended_batch_size,
            recommended_workers=recommended_workers,
            recommended_precision=recommended_precision
        )
        
        self._log_environment_info(env_info)
        return env_info
    
    def _recommend_batch_size(self, has_cuda: bool, memory_gb: float) -> int:
        """
        推荐批次大小
        
        Args:
            has_cuda: 是否有CUDA
            memory_gb: 可用内存（GB）
            
        Returns:
            int: 推荐的批次大小
        """
        if has_cuda:
            # GPU模式：根据显存推荐
            if torch.cuda.is_available():
                try:
                    # 获取第一个GPU的显存信息
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if gpu_memory_gb >= 24:  # 24GB+
                        return 4
                    elif gpu_memory_gb >= 16:  # 16GB+
                        return 3
                    elif gpu_memory_gb >= 8:  # 8GB+
                        return 2
                    else:
                        return 1
                except Exception:
                    return 2
            else:
                return 2
        else:
            # CPU模式：根据内存推荐
            if memory_gb >= 32:
                return 2
            elif memory_gb >= 16:
                return 1
            else:
                return 1
    
    def _recommend_workers(self, use_cpu: bool, cpu_count: int) -> int:
        """
        推荐工作进程数
        
        Args:
            use_cpu: 是否使用CPU
            cpu_count: CPU核心数
            
        Returns:
            int: 推荐的工作进程数
        """
        if use_cpu:
            # CPU模式：保守一些，避免过度占用
            return min(4, max(1, cpu_count // 2))
        else:
            # GPU模式：可以使用更多进程
            return min(8, max(1, cpu_count // 2))
    
    def _recommend_precision(self, has_cuda: bool) -> str:
        """
        推荐精度设置
        
        Args:
            has_cuda: 是否有CUDA
            
        Returns:
            str: 推荐的精度（fp32, fp16, bf16）
        """
        if has_cuda:
            # 检查是否支持bfloat16
            try:
                if torch.cuda.is_bf16_supported():
                    return "bf16"
                else:
                    return "fp16"
            except Exception:
                return "fp16"
        else:
            # CPU模式使用fp32
            return "fp32"
    
    def _setup_environment(self) -> None:
        """
        设置环境变量和优化选项
        """
        if self.env_info.use_cpu:
            logger.info("配置CPU优化环境...")
            
            # 设置CPU相关的环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # 优化CPU性能
            torch.set_num_threads(self.env_info.cpu_count)
            
            # 禁用一些GPU特定的功能
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # 内存优化
            if self.env_info.available_memory_gb < 16:
                logger.warning("可用内存较少，启用内存优化模式")
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        else:
            logger.info("配置GPU优化环境...")
            
            # GPU优化设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 设置内存分配策略
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    def _log_environment_info(self, env_info: EnvironmentInfo) -> None:
        """
        记录环境信息
        
        Args:
            env_info: 环境信息
        """
        logger.info("=== 环境信息 ===")
        logger.info(f"PyTorch版本: {env_info.pytorch_version}")
        logger.info(f"CUDA可用: {env_info.has_cuda}")
        if env_info.has_cuda:
            logger.info(f"CUDA版本: {env_info.cuda_version}")
            logger.info(f"GPU数量: {env_info.cuda_device_count}")
            for i in range(env_info.cuda_device_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                except Exception:
                    logger.info(f"GPU {i}: 信息获取失败")
        
        logger.info(f"CPU核心数: {env_info.cpu_count}")
        logger.info(f"可用内存: {env_info.available_memory_gb:.1f}GB")
        logger.info(f"使用CPU模式: {env_info.use_cpu}")
        logger.info(f"推荐批次大小: {env_info.recommended_batch_size}")
        logger.info(f"推荐工作进程数: {env_info.recommended_workers}")
        logger.info(f"推荐精度: {env_info.recommended_precision}")
        logger.info("==================")
    
    def adapt_training_args(self, training_args: Any) -> Any:
        """
        根据环境调整训练参数
        
        Args:
            training_args: 训练参数对象
            
        Returns:
            Any: 调整后的训练参数
        """
        logger.info("根据环境调整训练参数...")
        
        # 调整批次大小
        if hasattr(training_args, 'per_device_train_batch_size'):
            original_batch_size = training_args.per_device_train_batch_size
            if self.env_info.use_cpu and original_batch_size > self.env_info.recommended_batch_size:
                training_args.per_device_train_batch_size = self.env_info.recommended_batch_size
                logger.info(f"调整训练批次大小: {original_batch_size} -> {self.env_info.recommended_batch_size}")
        
        if hasattr(training_args, 'per_device_eval_batch_size'):
            original_eval_batch_size = training_args.per_device_eval_batch_size
            recommended_eval_batch_size = max(1, self.env_info.recommended_batch_size // 2)
            if self.env_info.use_cpu and original_eval_batch_size > recommended_eval_batch_size:
                training_args.per_device_eval_batch_size = recommended_eval_batch_size
                logger.info(f"调整评估批次大小: {original_eval_batch_size} -> {recommended_eval_batch_size}")
        
        # 调整数据加载器工作进程数
        if hasattr(training_args, 'dataloader_num_workers'):
            original_workers = training_args.dataloader_num_workers
            if original_workers > self.env_info.recommended_workers:
                training_args.dataloader_num_workers = self.env_info.recommended_workers
                logger.info(f"调整数据加载器工作进程数: {original_workers} -> {self.env_info.recommended_workers}")
        
        # 调整精度设置
        if self.env_info.use_cpu:
            # CPU模式强制使用fp32
            if hasattr(training_args, 'fp16'):
                training_args.fp16 = False
            if hasattr(training_args, 'bf16'):
                training_args.bf16 = False
            logger.info("CPU模式：禁用混合精度训练")
        else:
            # GPU模式根据推荐设置精度
            if self.env_info.recommended_precision == "bf16":
                if hasattr(training_args, 'bf16'):
                    training_args.bf16 = True
                if hasattr(training_args, 'fp16'):
                    training_args.fp16 = False
                logger.info("GPU模式：启用bfloat16精度")
            elif self.env_info.recommended_precision == "fp16":
                if hasattr(training_args, 'fp16'):
                    training_args.fp16 = True
                if hasattr(training_args, 'bf16'):
                    training_args.bf16 = False
                logger.info("GPU模式：启用float16精度")
        
        # CPU模式的特殊优化
        if self.env_info.use_cpu:
            # 禁用梯度检查点以节省内存
            if hasattr(training_args, 'gradient_checkpointing'):
                if training_args.gradient_checkpointing:
                    logger.info("CPU模式：禁用梯度检查点")
                    training_args.gradient_checkpointing = False
            
            # 调整保存策略
            if hasattr(training_args, 'save_steps'):
                if training_args.save_steps < 100:
                    training_args.save_steps = 100
                    logger.info("CPU模式：调整保存步数为100")
            
            # 调整评估策略
            if hasattr(training_args, 'eval_steps'):
                if training_args.eval_steps < 50:
                    training_args.eval_steps = 50
                    logger.info("CPU模式：调整评估步数为50")
        
        return training_args
    
    def adapt_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据环境调整模型配置
        
        Args:
            model_config: 模型配置字典
            
        Returns:
            Dict[str, Any]: 调整后的模型配置
        """
        logger.info("根据环境调整模型配置...")
        
        if self.env_info.use_cpu:
            # CPU模式的模型优化
            if 'torch_dtype' in model_config:
                model_config['torch_dtype'] = 'float32'
                logger.info("CPU模式：设置模型数据类型为float32")
            
            if 'use_flash_attention' in model_config:
                model_config['use_flash_attention'] = False
                logger.info("CPU模式：禁用Flash Attention")
            
            # 调整最大位置嵌入长度以节省内存
            if 'max_position_embeddings' in model_config:
                if model_config['max_position_embeddings'] > 8192:
                    model_config['max_position_embeddings'] = 8192
                    logger.info("CPU模式：限制最大位置嵌入长度为8192")
        
        return model_config
    
    def get_device_map(self) -> Optional[str]:
        """
        获取设备映射
        
        Returns:
            Optional[str]: 设备映射字符串
        """
        if self.env_info.use_cpu:
            return None  # CPU模式不需要设备映射
        else:
            if self.env_info.cuda_device_count == 1:
                return "auto"
            else:
                return "auto"  # 多GPU情况下让transformers自动处理
    
    def setup_distributed_training(self) -> Dict[str, Any]:
        """
        设置分布式训练环境
        
        Returns:
            Dict[str, Any]: 分布式训练配置
        """
        if self.env_info.use_cpu:
            logger.info("CPU模式：不支持分布式训练")
            return {"use_distributed": False}
        
        if self.env_info.cuda_device_count <= 1:
            logger.info("单GPU模式：不需要分布式训练")
            return {"use_distributed": False}
        
        logger.info(f"多GPU模式：设置分布式训练，GPU数量: {self.env_info.cuda_device_count}")
        return {
            "use_distributed": True,
            "world_size": self.env_info.cuda_device_count,
            "backend": "nccl"
        }
    
    def cleanup(self) -> None:
        """
        清理资源
        """
        if not self.env_info.use_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("清理GPU缓存")