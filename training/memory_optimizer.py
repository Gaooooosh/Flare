#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化模块

提供内存监控、优化和清理功能，特别针对CPU环境下的性能优化
"""

import gc
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class MemoryStats:
    """内存统计信息"""
    timestamp: float
    cpu_memory_used: float  # MB
    cpu_memory_total: float  # MB
    cpu_memory_percent: float
    gpu_memory_used: Optional[float] = None  # MB
    gpu_memory_total: Optional[float] = None  # MB
    gpu_memory_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history: List[MemoryStats] = []
        self._lock = threading.Lock()
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计信息"""
        # CPU内存信息
        cpu_memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            timestamp=time.time(),
            cpu_memory_used=cpu_memory.used / 1024 / 1024,  # 转换为MB
            cpu_memory_total=cpu_memory.total / 1024 / 1024,
            cpu_memory_percent=cpu_memory.percent
        )
        
        # GPU内存信息（如果可用）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                stats.gpu_memory_used = allocated / 1024 / 1024
                stats.gpu_memory_total = total / 1024 / 1024
                stats.gpu_memory_percent = (allocated / total) * 100 if total > 0 else 0
            except Exception:
                pass  # GPU内存获取失败时忽略
        
        return stats
    
    def start_monitoring(self):
        """开始内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                with self._lock:
                    self.memory_history.append(stats)
                    # 保持最近1000条记录
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]
                time.sleep(self.monitoring_interval)
            except Exception:
                pass  # 监控过程中的异常不应中断监控
    
    def get_monitoring_history(self) -> List[MemoryStats]:
        """获取监控历史"""
        with self._lock:
            return self.memory_history.copy()
    
    def cleanup_memory(self):
        """清理内存"""
        # Python垃圾回收
        gc.collect()
        
        # PyTorch内存清理
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, use_cpu: bool = False, monitoring_interval: float = 1.0):
        self.use_cpu = use_cpu
        self.monitor = MemoryMonitor(monitoring_interval)
        self.optimization_history: List[Dict[str, Any]] = []
    
    def get_system_memory_info(self) -> Dict[str, Any]:
        """获取系统内存信息"""
        cpu_memory = psutil.virtual_memory()
        info = {
            'cpu_memory_total_gb': cpu_memory.total / 1024 / 1024 / 1024,
            'cpu_memory_available_gb': cpu_memory.available / 1024 / 1024 / 1024,
            'cpu_cores': psutil.cpu_count(),
            'cpu_cores_physical': psutil.cpu_count(logical=False)
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available() and not self.use_cpu:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                info.update({
                    'gpu_memory_total_gb': gpu_props.total_memory / 1024 / 1024 / 1024,
                    'gpu_name': gpu_props.name,
                    'gpu_compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })
            except Exception:
                pass
        
        return info
    
    def optimize_for_training(self) -> Dict[str, Any]:
        """为训练优化内存使用"""
        system_info = self.get_system_memory_info()
        optimizations = {}
        
        if self.use_cpu:
            # CPU模式优化
            cpu_memory_gb = system_info['cpu_memory_available_gb']
            cpu_cores = system_info['cpu_cores_physical']
            
            # 基于可用内存调整批次大小
            if cpu_memory_gb < 4:
                batch_size = 1
                gradient_accumulation = 8
                num_workers = 0
            elif cpu_memory_gb < 8:
                batch_size = 2
                gradient_accumulation = 4
                num_workers = min(2, cpu_cores)
            elif cpu_memory_gb < 16:
                batch_size = 4
                gradient_accumulation = 2
                num_workers = min(4, cpu_cores)
            else:
                # 即使在大内存系统上，CPU模式也应该使用较小的批次大小
                batch_size = 4
                gradient_accumulation = 2
                num_workers = min(2, max(1, cpu_cores // 8))  # 更保守的工作进程数
            
            optimizations.update({
                'recommended_batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation,
                'dataloader_num_workers': num_workers,
                'fp16': False,  # CPU不支持fp16
                'bf16': False,  # CPU通常不支持bf16
                'gradient_checkpointing': True,  # 启用梯度检查点节省内存
                'optim': 'adamw_torch',  # 使用PyTorch原生优化器
                'dataloader_pin_memory': False,  # CPU模式不需要pin memory
            })
        else:
            # GPU模式优化
            gpu_memory_gb = system_info.get('gpu_memory_total_gb', 8)
            
            if gpu_memory_gb < 6:
                batch_size = 2
                gradient_accumulation = 8
            elif gpu_memory_gb < 12:
                batch_size = 4
                gradient_accumulation = 4
            elif gpu_memory_gb < 24:
                batch_size = 8
                gradient_accumulation = 2
            else:
                batch_size = 16
                gradient_accumulation = 1
            
            optimizations.update({
                'recommended_batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation,
                'dataloader_num_workers': 4,
                'fp16': True,  # 启用混合精度
                'gradient_checkpointing': gpu_memory_gb < 12,  # 小显存启用梯度检查点
                'dataloader_pin_memory': True,
            })
        
        # 记录优化历史
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'optimizations': optimizations,
            'use_cpu': self.use_cpu
        }
        self.optimization_history.append(optimization_record)
        
        return optimizations
    
    def start_monitoring(self):
        """开始内存监控"""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitor.stop_monitoring()
    
    def cleanup_memory(self):
        """清理内存"""
        self.monitor.cleanup_memory()
    
    def get_memory_stats(self) -> MemoryStats:
        """获取当前内存统计"""
        return self.monitor.get_memory_stats()
    
    def save_memory_report(self, output_dir: str):
        """保存内存使用报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取监控历史
        history = self.monitor.get_monitoring_history()
        
        # 生成报告
        report = {
            'generation_time': datetime.now().isoformat(),
            'system_info': self.get_system_memory_info(),
            'use_cpu': self.use_cpu,
            'optimization_history': self.optimization_history,
            'memory_monitoring': {
                'total_records': len(history),
                'monitoring_duration_minutes': (history[-1].timestamp - history[0].timestamp) / 60 if len(history) > 1 else 0,
                'peak_cpu_memory_mb': max(stats.cpu_memory_used for stats in history) if history else 0,
                'average_cpu_memory_mb': sum(stats.cpu_memory_used for stats in history) / len(history) if history else 0,
            }
        }
        
        # 添加GPU内存统计（如果可用）
        gpu_stats = [stats for stats in history if stats.gpu_memory_used is not None]
        if gpu_stats:
            report['memory_monitoring'].update({
                'peak_gpu_memory_mb': max(stats.gpu_memory_used for stats in gpu_stats),
                'average_gpu_memory_mb': sum(stats.gpu_memory_used for stats in gpu_stats) / len(gpu_stats),
            })
        
        # 保存详细历史（最近100条记录）
        if history:
            report['detailed_history'] = [stats.to_dict() for stats in history[-100:]]
        
        # 写入文件
        report_file = output_path / 'memory_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def get_memory_recommendations(self) -> List[str]:
        """获取内存优化建议"""
        current_stats = self.get_memory_stats()
        recommendations = []
        
        # CPU内存建议
        if current_stats.cpu_memory_percent > 90:
            recommendations.append("CPU内存使用率过高，建议减少批次大小或启用梯度检查点")
        elif current_stats.cpu_memory_percent > 80:
            recommendations.append("CPU内存使用率较高，建议监控内存使用情况")
        
        # GPU内存建议
        if current_stats.gpu_memory_percent and current_stats.gpu_memory_percent > 90:
            recommendations.append("GPU内存使用率过高，建议减少批次大小或启用梯度检查点")
        elif current_stats.gpu_memory_percent and current_stats.gpu_memory_percent > 80:
            recommendations.append("GPU内存使用率较高，建议启用混合精度训练")
        
        # 通用建议
        if self.use_cpu:
            recommendations.extend([
                "CPU模式下建议使用较小的批次大小和更多的梯度累积步数",
                "考虑使用梯度检查点来节省内存",
                "定期执行垃圾回收以释放未使用的内存"
            ])
        
        return recommendations