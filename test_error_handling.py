#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理机制测试脚本

测试内容：
1. 错误处理器的基本功能
2. 不同类型错误的处理
3. 错误恢复机制
4. 错误日志记录
5. 错误报告生成
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/xiaoyonggao/Flare')
sys.path.insert(0, '/home/xiaoyonggao/Flare/training')

from training.error_handler import (
    ErrorHandler, TrainingError, DatasetError, ModelError, 
    EnvironmentError, MemoryError, ErrorSeverity,
    error_handler_decorator, safe_execute, get_global_error_handler
)

def test_error_handler_basic():
    """测试错误处理器基本功能"""
    print("\n=== 测试错误处理器基本功能 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        # 测试错误处理
        error = TrainingError("测试错误", severity=ErrorSeverity.WARNING)
        result = handler.handle_error(error)
        
        # 检查日志文件是否创建
        assert log_file.exists(), "错误日志文件未创建"
        
        # 检查错误摘要
        summary = handler.get_error_summary()
        assert summary['total_errors'] == 1, f"错误计数不正确: {summary['total_errors']}"
        assert summary['by_severity']['warning'] == 1, "警告严重程度错误计数不正确"
        
        print("✓ 错误处理器基本功能测试通过")
        return True

def test_different_error_types():
    """测试不同类型的错误处理"""
    print("\n=== 测试不同类型错误处理 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        # 测试数据集错误
        dataset_error = DatasetError(
            "数据集加载失败", 
            dataset_name="test_dataset",
            severity=ErrorSeverity.ERROR
        )
        handler.handle_error(dataset_error)
        
        # 测试模型错误
        model_error = ModelError(
            "模型加载失败",
            model_name="test_model",
            severity=ErrorSeverity.CRITICAL
        )
        handler.handle_error(model_error)
        
        # 测试环境错误
        env_error = EnvironmentError(
            "GPU不可用",
            environment_info={"gpu_count": 0},
            severity=ErrorSeverity.WARNING
        )
        handler.handle_error(env_error)
        
        # 测试内存错误
        memory_error = MemoryError(
            "内存不足",
            memory_info={"available_gb": 2},
            severity=ErrorSeverity.ERROR
        )
        handler.handle_error(memory_error)
        
        # 检查错误摘要
        summary = handler.get_error_summary()
        assert summary['total_errors'] == 4, f"总错误数不正确: {summary['total_errors']}"
        assert summary['by_type']['DATASET_ERROR'] == 1, "数据集错误计数不正确"
        assert summary['by_type']['MODEL_ERROR'] == 1, "模型错误计数不正确"
        assert summary['by_type']['ENVIRONMENT_ERROR'] == 1, "环境错误计数不正确"
        assert summary['by_type']['MEMORY_ERROR'] == 1, "内存错误计数不正确"
        
        print("✓ 不同类型错误处理测试通过")
        return True

def test_error_recovery():
    """测试错误恢复机制"""
    print("\n=== 测试错误恢复机制 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        # 注册恢复策略
        def memory_recovery_strategy(error):
            print(f"执行内存恢复策略: {error.message}")
            return True
        
        handler.register_recovery_strategy(MemoryError, memory_recovery_strategy)
        
        # 测试恢复
        memory_error = MemoryError(
            "内存不足",
            memory_info={"available_gb": 1},
            severity=ErrorSeverity.ERROR
        )
        
        result = handler.handle_error(memory_error)
        assert result, "错误恢复失败"
        
        print("✓ 错误恢复机制测试通过")
        return True

def test_error_decorator():
    """测试错误处理装饰器"""
    print("\n=== 测试错误处理装饰器 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        @error_handler_decorator(handler, reraise=False)
        def test_function():
            raise ValueError("测试异常")
        
        # 执行函数，应该捕获异常
        result = test_function()
        assert result is None, "装饰器应该返回None"
        
        # 检查错误是否被记录
        summary = handler.get_error_summary()
        assert summary['total_errors'] == 1, "装饰器未正确记录错误"
        
        print("✓ 错误处理装饰器测试通过")
        return True

def test_safe_execute():
    """测试安全执行函数"""
    print("\n=== 测试安全执行函数 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        def failing_function():
            raise RuntimeError("测试运行时错误")
        
        def success_function():
            return "成功"
        
        # 测试失败的函数
        result = safe_execute(failing_function, handler, default_value="默认值")
        assert result == "默认值", f"安全执行返回值不正确: {result}"
        
        # 测试成功的函数
        result = safe_execute(success_function, handler)
        assert result == "成功", f"安全执行返回值不正确: {result}"
        
        print("✓ 安全执行函数测试通过")
        return True

def test_error_report_generation():
    """测试错误报告生成"""
    print("\n=== 测试错误报告生成 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_error.log"
        report_file = Path(temp_dir) / "error_report.json"
        handler = ErrorHandler(log_file=str(log_file), enable_recovery=True)
        
        # 添加一些错误
        handler.handle_error(DatasetError("数据集错误1", dataset_name="test1"))
        handler.handle_error(ModelError("模型错误1", model_name="test_model"))
        handler.handle_error(MemoryError("内存错误1", memory_info={"available_gb": 1}))
        
        # 生成报告
        handler.save_error_report(str(report_file))
        
        # 检查报告文件
        assert report_file.exists(), "错误报告文件未创建"
        
        # 读取并验证报告内容
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        assert 'summary' in report, "报告缺少摘要信息"
        assert 'errors' in report, "报告缺少错误详情"
        assert report['summary']['total_errors'] == 3, "报告中错误总数不正确"
        assert len(report['errors']) == 3, "报告中错误详情数量不正确"
        
        print("✓ 错误报告生成测试通过")
        return True

def run_all_tests():
    """运行所有测试"""
    print("开始错误处理机制测试...")
    
    tests = [
        test_error_handler_basic,
        test_different_error_types,
        test_error_recovery,
        test_error_decorator,
        test_safe_execute,
        test_error_report_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test.__name__} 失败: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有错误处理测试通过！")
        return True
    else:
        print("❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)