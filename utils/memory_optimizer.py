#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化器模块

此模块提供内存监控和优化功能，特别针对CPU环境下的性能问题。
包括内存使用监控、垃圾回收优化、批次大小动态调整等功能。
"""

import os
import gc
import psutil
import torch
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    gpu_memory_gb: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, log_interval: int = 60, enable_gpu_monitoring: bool = True):
        """
        初始化内存监控器
        
        Args:
            log_interval: 日志记录间隔（秒）
            enable_gpu_monitoring: 是否启用GPU内存监控
        """
        self.log_interval = log_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history = []
        self.max_history_size = 1000
        
    def get_current_stats(self) -> MemoryStats:
        """获取当前内存统计信息"""
        # CPU内存统计
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        used_memory_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        # GPU内存统计
        gpu_memory_gb = None
        gpu_memory_used_gb = None
        gpu_memory_percent = None
        
        if self.enable_gpu_monitoring:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                gpu_memory_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_percent = (gpu_memory_used_gb / gpu_memory_gb) * 100
            except Exception as e:
                logger.warning(f"GPU内存监控失败: {e}")
        
        return MemoryStats(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            used_memory_gb=used_memory_gb,
            memory_percent=memory_percent,
            gpu_memory_gb=gpu_memory_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_percent=gpu_memory_percent
        )
    
    def start_monitoring(self):
        """开始内存监控"""
        if self.monitoring:
            logger.warning("内存监控已在运行")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"内存监控已启动，日志间隔: {self.log_interval}秒")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("内存监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                stats = self.get_current_stats()
                self._log_stats(stats)
                self._update_history(stats)
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(self.log_interval)
    
    def _log_stats(self, stats: MemoryStats):
        """记录内存统计信息"""
        log_msg = (
            f"内存使用: {stats.used_memory_gb:.1f}GB/{stats.total_memory_gb:.1f}GB "
            f"({stats.memory_percent:.1f}%), 可用: {stats.available_memory_gb:.1f}GB"
        )
        
        if stats.gpu_memory_gb is not None:
            log_msg += (
                f", GPU内存: {stats.gpu_memory_used_gb:.1f}GB/{stats.gpu_memory_gb:.1f}GB "
                f"({stats.gpu_memory_percent:.1f}%)"
            )
        
        logger.info(log_msg)
    
    def _update_history(self, stats: MemoryStats):
        """更新历史记录"""
        self.stats_history.append(stats)
        if len(self.stats_history) > self.max_history_size:
            self.stats_history.pop(0)
    
    def get_peak_usage(self) -> Optional[MemoryStats]:
        """获取峰值内存使用"""
        if not self.stats_history:
            return None
        
        return max(self.stats_history, key=lambda x: x.memory_percent)
    
    def get_average_usage(self) -> Optional[Dict[str, float]]:
        """获取平均内存使用"""
        if not self.stats_history:
            return None
        
        avg_memory_percent = sum(s.memory_percent for s in self.stats_history) / len(self.stats_history)
        avg_used_gb = sum(s.used_memory_gb for s in self.stats_history) / len(self.stats_history)
        
        result = {
            "avg_memory_percent": avg_memory_percent,
            "avg_used_gb": avg_used_gb
        }
        
        if self.enable_gpu_monitoring and any(s.gpu_memory_percent for s in self.stats_history):
            gpu_stats = [s for s in self.stats_history if s.gpu_memory_percent is not None]
            if gpu_stats:
                result["avg_gpu_memory_percent"] = sum(s.gpu_memory_percent for s in gpu_stats) / len(gpu_stats)
                result["avg_gpu_used_gb"] = sum(s.gpu_memory_used_gb for s in gpu_stats) / len(gpu_stats)
        
        return result


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, use_cpu: bool = False, aggressive_gc: bool = True):
        """
        初始化内存优化器
        
        Args:
            use_cpu: 是否使用CPU模式
            aggressive_gc: 是否启用激进的垃圾回收
        """
        self.use_cpu = use_cpu
        self.aggressive_gc = aggressive_gc
        self.monitor = MemoryMonitor(enable_gpu_monitoring=not use_cpu)
        
    def optimize_for_training(self) -> Dict[str, Any]:
        """为训练优化内存设置"""
        optimizations = {}
        
        # 获取当前内存状态
        stats = self.monitor.get_current_stats()
        logger.info(f"当前内存使用: {stats.memory_percent:.1f}%")
        
        # CPU模式优化
        if self.use_cpu:
            optimizations.update(self._optimize_cpu_mode(stats))
        else:
            optimizations.update(self._optimize_gpu_mode(stats))
        
        # 通用优化
        optimizations.update(self._apply_general_optimizations())
        
        return optimizations
    
    def _optimize_cpu_mode(self, stats: MemoryStats) -> Dict[str, Any]:
        """CPU模式优化"""
        optimizations = {}
        
        # 根据可用内存调整批次大小
        if stats.available_memory_gb < 4:
            optimizations["recommended_batch_size"] = 1
            optimizations["gradient_accumulation_steps"] = 8
            logger.warning("内存不足，建议使用小批次大小")
        elif stats.available_memory_gb < 8:
            optimizations["recommended_batch_size"] = 2
            optimizations["gradient_accumulation_steps"] = 4
        else:
            optimizations["recommended_batch_size"] = 4
            optimizations["gradient_accumulation_steps"] = 2
        
        # CPU特定优化
        optimizations["dataloader_num_workers"] = min(4, os.cpu_count() // 2)
        optimizations["pin_memory"] = False
        optimizations["torch_dtype"] = "float32"  # CPU模式使用float32
        
        return optimizations
    
    def _optimize_gpu_mode(self, stats: MemoryStats) -> Dict[str, Any]:
        """GPU模式优化"""
        optimizations = {}
        
        if stats.gpu_memory_gb is not None:
            # 根据GPU内存调整批次大小
            if stats.gpu_memory_gb < 8:
                optimizations["recommended_batch_size"] = 1
                optimizations["gradient_accumulation_steps"] = 8
            elif stats.gpu_memory_gb < 16:
                optimizations["recommended_batch_size"] = 2
                optimizations["gradient_accumulation_steps"] = 4
            else:
                optimizations["recommended_batch_size"] = 4
                optimizations["gradient_accumulation_steps"] = 2
        
        # GPU特定优化
        optimizations["dataloader_num_workers"] = 4
        optimizations["pin_memory"] = True
        optimizations["torch_dtype"] = "bfloat16"
        
        return optimizations
    
    def _apply_general_optimizations(self) -> Dict[str, Any]:
        """应用通用优化"""
        optimizations = {}
        
        # 启用梯度检查点以节省内存
        optimizations["gradient_checkpointing"] = True
        
        # 优化数据加载
        optimizations["dataloader_drop_last"] = True
        optimizations["remove_unused_columns"] = True
        
        # 设置垃圾回收
        if self.aggressive_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizations["gc_enabled"] = True
        
        return optimizations
    
    def cleanup_memory(self):
        """清理内存"""
        logger.info("执行内存清理...")
        
        # Python垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收清理了 {collected} 个对象")
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
        
        # 记录清理后的内存状态
        stats = self.monitor.get_current_stats()
        logger.info(f"清理后内存使用: {stats.memory_percent:.1f}%")
    
    def start_monitoring(self):
        """开始内存监控"""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitor.stop_monitoring()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """生成内存使用报告"""
        current_stats = self.monitor.get_current_stats()
        peak_stats = self.monitor.get_peak_usage()
        avg_stats = self.monitor.get_average_usage()
        
        report = {
            "current": {
                "memory_percent": current_stats.memory_percent,
                "used_gb": current_stats.used_memory_gb,
                "available_gb": current_stats.available_memory_gb
            }
        }
        
        if peak_stats:
            report["peak"] = {
                "memory_percent": peak_stats.memory_percent,
                "used_gb": peak_stats.used_memory_gb
            }
        
        if avg_stats:
            report["average"] = avg_stats
        
        if current_stats.gpu_memory_gb is not None:
            report["current"]["gpu_memory_percent"] = current_stats.gpu_memory_percent
            report["current"]["gpu_used_gb"] = current_stats.gpu_memory_used_gb
        
        return report
    
    def save_memory_report(self, output_dir: str):
        """保存内存使用报告"""
        report = self.get_memory_report()
        output_path = Path(output_dir) / "memory_report.json"
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"内存报告已保存到: {output_path}")


def get_optimal_batch_size(available_memory_gb: float, 
                          model_size_gb: float = 3.0,
                          use_cpu: bool = False) -> Tuple[int, int]:
    """
    根据可用内存计算最优批次大小
    
    Args:
        available_memory_gb: 可用内存（GB）
        model_size_gb: 模型大小（GB）
        use_cpu: 是否使用CPU模式
        
    Returns:
        Tuple[int, int]: (batch_size, gradient_accumulation_steps)
    """
    # 预留内存用于其他操作
    usable_memory = available_memory_gb * 0.7
    
    # CPU模式需要更多内存用于计算
    if use_cpu:
        memory_per_sample = model_size_gb * 0.5  # CPU模式每个样本需要更多内存
    else:
        memory_per_sample = model_size_gb * 0.2  # GPU模式相对高效
    
    # 计算最大批次大小
    max_batch_size = max(1, int(usable_memory / memory_per_sample))
    
    # 限制批次大小范围
    if max_batch_size >= 8:
        return 8, 1
    elif max_batch_size >= 4:
        return 4, 2
    elif max_batch_size >= 2:
        return 2, 4
    else:
        return 1, 8