#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理模块

提供统一的错误处理、日志记录和异常恢复机制
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
from functools import wraps
from enum import Enum

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"

class TrainingError(Exception):
    """训练相关的基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "TRAINING_ERROR"
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()

class DatasetError(TrainingError):
    """数据集相关错误"""
    
    def __init__(self, message: str, dataset_name: str = None, **kwargs):
        super().__init__(message, error_code="DATASET_ERROR", **kwargs)
        if dataset_name:
            self.context['dataset_name'] = dataset_name

class ModelError(TrainingError):
    """模型相关错误"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        if model_name:
            self.context['model_name'] = model_name

class EnvironmentError(TrainingError):
    """环境相关错误"""
    
    def __init__(self, message: str, environment_info: Dict[str, Any] = None, **kwargs):
        super().__init__(message, error_code="ENVIRONMENT_ERROR", **kwargs)
        if environment_info:
            self.context.update(environment_info)

class MemoryError(TrainingError):
    """内存相关错误"""
    
    def __init__(self, message: str, memory_info: Dict[str, Any] = None, **kwargs):
        super().__init__(message, error_code="MEMORY_ERROR", **kwargs)
        if memory_info:
            self.context.update(memory_info)

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, log_file: Optional[str] = None, enable_recovery: bool = True):
        self.log_file = log_file
        self.enable_recovery = enable_recovery
        self.error_history = []
        self.recovery_strategies = {}
        self.logger = self._setup_logger()
        
        # 注册默认恢复策略
        self._register_default_recovery_strategies()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('error_handler')
        logger.setLevel(logging.DEBUG)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _register_default_recovery_strategies(self):
        """注册默认的恢复策略"""
        
        def dataset_fallback_strategy(error: DatasetError) -> bool:
            """数据集回退策略"""
            self.logger.info("尝试数据集回退策略...")
            # 这里可以实现具体的回退逻辑
            return False
        
        def memory_cleanup_strategy(error: MemoryError) -> bool:
            """内存清理策略"""
            self.logger.info("执行内存清理策略...")
            try:
                import gc
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                
                self.logger.info("内存清理完成")
                return True
            except Exception as e:
                self.logger.error(f"内存清理失败: {e}")
                return False
        
        def environment_check_strategy(error: EnvironmentError) -> bool:
            """环境检查策略"""
            self.logger.info("执行环境检查策略...")
            # 这里可以实现环境检查和修复逻辑
            return False
        
        self.recovery_strategies.update({
            'DATASET_ERROR': dataset_fallback_strategy,
            'MEMORY_ERROR': memory_cleanup_strategy,
            'ENVIRONMENT_ERROR': environment_check_strategy
        })
    
    def register_recovery_strategy(self, error_code: str, strategy: Callable[[TrainingError], bool]):
        """注册恢复策略"""
        self.recovery_strategies[error_code] = strategy
        self.logger.debug(f"注册恢复策略: {error_code}")
    
    def handle_error(self, error: Union[Exception, TrainingError], context: Dict[str, Any] = None) -> bool:
        """处理错误"""
        # 转换为TrainingError
        if not isinstance(error, TrainingError):
            training_error = TrainingError(
                message=str(error),
                error_code="UNKNOWN_ERROR",
                severity=ErrorSeverity.MEDIUM,
                context=context or {}
            )
            training_error.__cause__ = error
        else:
            training_error = error
            if context:
                training_error.context.update(context)
        
        # 记录错误
        self._log_error(training_error)
        
        # 添加到历史记录
        self.error_history.append({
            'timestamp': training_error.timestamp,
            'error_code': training_error.error_code,
            'message': training_error.message,
            'severity': training_error.severity.value,
            'context': training_error.context,
            'traceback': traceback.format_exc() if hasattr(training_error, '__cause__') else None
        })
        
        # 尝试恢复
        if self.enable_recovery:
            return self._attempt_recovery(training_error)
        
        return False
    
    def _log_error(self, error: TrainingError):
        """记录错误日志"""
        log_message = f"[{error.error_code}] {error.message}"
        
        if error.context:
            log_message += f" | Context: {json.dumps(error.context, ensure_ascii=False)}"
        
        # 根据严重程度选择日志级别
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_recovery(self, error: TrainingError) -> bool:
        """尝试错误恢复"""
        strategy = self.recovery_strategies.get(error.error_code)
        if strategy:
            try:
                self.logger.info(f"尝试恢复策略: {error.error_code}")
                success = strategy(error)
                if success:
                    self.logger.info(f"恢复成功: {error.error_code}")
                else:
                    self.logger.warning(f"恢复失败: {error.error_code}")
                return success
            except Exception as e:
                self.logger.error(f"恢复策略执行异常: {e}")
                return False
        else:
            self.logger.debug(f"没有找到恢复策略: {error.error_code}")
            return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}, 'severity_distribution': {}}
        
        error_types = {}
        severity_distribution = {}
        
        for error in self.error_history:
            # 统计错误类型
            error_code = error['error_code']
            error_types[error_code] = error_types.get(error_code, 0) + 1
            
            # 统计严重程度分布
            severity = error['severity']
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'severity_distribution': severity_distribution,
            'by_type': error_types,  # 为了兼容测试
            'by_severity': severity_distribution,  # 为了兼容测试
            'recent_errors': self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }
    
    def save_error_report(self, output_path: str):
        """保存错误报告"""
        output_file = Path(output_path)
        
        # 如果传入的是目录，则在目录下创建error_report.json
        if output_file.suffix != '.json':
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / 'error_report.json'
        else:
            # 如果传入的是文件路径，确保父目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': self.get_error_summary(),
            'errors': self.error_history
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"错误报告已保存到: {output_file}")

def error_handler_decorator(handler: ErrorHandler, reraise: bool = True, return_value: Any = None):
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # 限制长度
                    'kwargs': str(kwargs)[:200]
                }
                
                recovered = handler.handle_error(e, context)
                
                if not recovered and reraise:
                    raise
                
                return return_value
        return wrapper
    return decorator

def safe_execute(func: Callable, error_handler: ErrorHandler, *args, default_value: Any = None, **kwargs) -> Any:
    """安全执行函数"""
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as e:
        context = {
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'args': str(args)[:200],
            'kwargs': str({k: v for k, v in kwargs.items() if k != 'default_value'})[:200]
        }
        
        recovered = error_handler.handle_error(e, context)
        return default_value

# 全局错误处理器实例
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def set_global_error_handler(handler: ErrorHandler):
    """设置全局错误处理器"""
    global _global_error_handler
    _global_error_handler = handler