#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化功能测试脚本

测试内存监控、优化建议和清理功能
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from training.memory_optimizer import MemoryOptimizer, MemoryMonitor
from training.environment_adapter import EnvironmentAdapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_memory_monitor():
    """测试内存监控功能"""
    logger.info("=== 测试内存监控功能 ===")
    
    monitor = MemoryMonitor()
    
    # 获取当前内存状态
    stats = monitor.get_memory_stats()
    logger.info(f"当前内存状态: {stats}")
    
    # 测试内存监控
    monitor.start_monitoring()
    
    # 模拟一些内存使用
    logger.info("模拟内存使用...")
    data = [torch.randn(1000, 1000) for _ in range(10)]
    time.sleep(2)
    
    # 清理内存
    del data
    monitor.cleanup_memory()
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 获取监控历史
    history = monitor.get_monitoring_history()
    logger.info(f"监控历史记录数: {len(history)}")
    
    return True

def test_memory_optimizer():
    """测试内存优化器功能"""
    logger.info("=== 测试内存优化器功能 ===")
    
    # 测试CPU模式
    logger.info("测试CPU模式优化...")
    optimizer_cpu = MemoryOptimizer(use_cpu=True)
    optimizations_cpu = optimizer_cpu.optimize_for_training()
    logger.info(f"CPU模式优化建议: {optimizations_cpu}")
    
    # 测试GPU模式（如果可用）
    if torch.cuda.is_available():
        logger.info("测试GPU模式优化...")
        optimizer_gpu = MemoryOptimizer(use_cpu=False)
        optimizations_gpu = optimizer_gpu.optimize_for_training()
        logger.info(f"GPU模式优化建议: {optimizations_gpu}")
    else:
        logger.info("GPU不可用，跳过GPU模式测试")
    
    return True

def test_memory_optimization_with_training_simulation():
    """模拟训练过程中的内存优化"""
    logger.info("=== 模拟训练过程内存优化 ===")
    
    # 初始化环境适配器
    env_adapter = EnvironmentAdapter()
    
    # 初始化内存优化器
    optimizer = MemoryOptimizer(use_cpu=env_adapter.env_info.use_cpu)
    
    # 获取优化建议
    optimizations = optimizer.optimize_for_training()
    logger.info(f"训练优化建议: {optimizations}")
    
    # 启动监控
    optimizer.start_monitoring()
    
    try:
        # 模拟训练步骤
        logger.info("模拟训练步骤...")
        for step in range(5):
            logger.info(f"训练步骤 {step + 1}")
            
            # 模拟模型前向传播
            if env_adapter.env_info.use_cpu:
                data = torch.randn(2, 512, 768)  # CPU模式使用较小的数据
            else:
                data = torch.randn(8, 512, 768)  # GPU模式可以使用更大的数据
            
            # 模拟一些计算
            result = torch.matmul(data, data.transpose(-2, -1))
            
            # 每步清理内存
            if step % 2 == 0:
                optimizer.cleanup_memory()
            
            time.sleep(1)
            
            # 清理临时变量
            del data, result
    
    finally:
        # 停止监控
        optimizer.stop_monitoring()
    
    # 保存内存报告
    output_dir = Path("./test_output")
    output_dir.mkdir(exist_ok=True)
    optimizer.save_memory_report(str(output_dir))
    
    logger.info(f"内存报告已保存到: {output_dir / 'memory_report.json'}")
    
    return True

def test_environment_integration():
    """测试与环境适配器的集成"""
    logger.info("=== 测试环境集成 ===")
    
    # 测试不同环境下的内存优化
    environments = [
        {'force_cpu': True, 'name': 'CPU强制模式'},
        {'force_cpu': False, 'name': '自动检测模式'}
    ]
    
    for env_config in environments:
        logger.info(f"测试 {env_config['name']}...")
        
        env_adapter = EnvironmentAdapter(force_cpu=env_config['force_cpu'])
        optimizer = MemoryOptimizer(use_cpu=env_adapter.env_info.use_cpu)
        
        optimizations = optimizer.optimize_for_training()
        logger.info(f"{env_config['name']} 优化建议: {optimizations}")
        
        # 验证优化建议的合理性
        if env_adapter.env_info.use_cpu:
            assert optimizations['recommended_batch_size'] <= 4, "CPU模式批次大小应该较小"
            assert optimizations['dataloader_num_workers'] <= 2, "CPU模式工作进程应该较少"
        
        logger.info(f"{env_config['name']} 测试通过")
    
    return True

def main():
    """主测试函数"""
    logger.info("开始内存优化功能测试")
    
    tests = [
        ("内存监控", test_memory_monitor),
        ("内存优化器", test_memory_optimizer),
        ("训练模拟", test_memory_optimization_with_training_simulation),
        ("环境集成", test_environment_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n开始测试: {test_name}")
            result = test_func()
            if result:
                logger.info(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 测试失败")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
            failed += 1
    
    # 测试结果汇总
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    logger.info(f"\n=== 测试结果汇总 ===")
    logger.info(f"总测试数: {total}")
    logger.info(f"通过: {passed}")
    logger.info(f"失败: {failed}")
    logger.info(f"成功率: {success_rate:.1f}%")
    
    if failed > 0:
        logger.error("部分测试失败，请检查日志")
        sys.exit(1)
    else:
        logger.info("所有测试通过！")
        sys.exit(0)

if __name__ == "__main__":
    main()