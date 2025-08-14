#!/usr/bin/env python3
"""
简化版训练系统测试脚本
验证基本功能而不进行实际训练
"""

import torch
import sys
import os
from pathlib import Path

def test_gpu_setup():
    """测试GPU设置功能"""
    print("🔧 测试GPU设置功能")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    total_gpus = torch.cuda.device_count()
    print(f"✅ 检测到 {total_gpus} 个GPU")
    
    # 显示GPU信息
    for i in range(total_gpus):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
        except Exception as e:
            print(f"   GPU {i}: 获取信息失败 - {e}")
    
    return True

def test_gpu_selection(gpu_ids):
    """测试GPU选择功能"""
    print(f"\n🎯 测试GPU选择: {gpu_ids}")
    print("=" * 40)
    
    total_gpus = torch.cuda.device_count()
    
    # 验证GPU ID
    invalid_ids = [gid for gid in gpu_ids if gid >= total_gpus or gid < 0]
    if invalid_ids:
        print(f"❌ 无效的GPU ID: {invalid_ids}")
        print(f"   可用GPU ID: 0-{total_gpus-1}")
        return False
    
    print(f"✅ GPU ID验证通过: {gpu_ids}")
    
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"✅ 设置CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    return True

def test_imports():
    """测试必要的导入"""
    print("\n📦 测试模块导入")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ transformers导入成功")
    except ImportError as e:
        print(f"❌ transformers导入失败: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ datasets导入成功")
    except ImportError as e:
        print(f"❌ datasets导入失败: {e}")
        return False
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        from patch_qwen_rope import patch_qwen_rope
        print("✅ patch_qwen_rope导入成功")
    except ImportError as e:
        print(f"❌ patch_qwen_rope导入失败: {e}")
        return False
    
    return True

def test_files():
    """测试必要文件是否存在"""
    print("\n📁 测试文件存在性")
    print("=" * 40)
    
    required_files = [
        "../training/train_qwen_simple.py",
        "../scripts/run_training_simple.sh",
        "../configs/training_config_simple.json",
        "../utils/patch_qwen_rope.py",
        "../docs/README_SIMPLE.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def test_config_file():
    """测试配置文件"""
    print("\n⚙️  测试配置文件")
    print("=" * 40)
    
    config_file = "../configs/training_config_simple.json"
    
    if not Path(config_file).exists():
        print(f"❌ 配置文件 {config_file} 不存在")
        return False
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("✅ 配置文件JSON格式正确")
        
        # 检查关键配置
        required_sections = ["model_args", "data_args", "training_args", "gpu_examples"]
        for section in required_sections:
            if section in config:
                print(f"✅ 配置节 '{section}' 存在")
            else:
                print(f"❌ 配置节 '{section}' 缺失")
                return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return False

def test_script_syntax():
    """测试脚本语法"""
    print("\n🐍 测试Python脚本语法")
    print("=" * 40)
    
    script_file = "../training/train_qwen_simple.py"
    
    if not Path(script_file).exists():
        print(f"❌ 脚本文件 {script_file} 不存在")
        return False
    
    try:
        import ast
        with open(script_file, 'r') as f:
            content = f.read()
        
        ast.parse(content)
        print(f"✅ {script_file} 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"❌ {script_file} 语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查 {script_file} 失败: {e}")
        return False

def main():
    """主函数"""
    print("🧪 简化版训练系统测试")
    print("=" * 60)
    
    tests = [
        ("GPU环境", test_gpu_setup),
        ("模块导入", test_imports),
        ("文件存在性", test_files),
        ("配置文件", test_config_file),
        ("脚本语法", test_script_syntax),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # GPU选择测试
    if torch.cuda.is_available():
        gpu_test_cases = [
            [0],
            [0, 1],
            [0, 1, 2, 3],
        ]
        
        for gpu_ids in gpu_test_cases:
            try:
                result = test_gpu_selection(gpu_ids)
                results.append((f"GPU选择 {gpu_ids}", result))
            except Exception as e:
                print(f"❌ GPU选择 {gpu_ids} 测试异常: {e}")
                results.append((f"GPU选择 {gpu_ids}", False))
    
    # 生成报告
    print("\n\n📊 测试报告")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
    
    print(f"\n📈 总结:")
    print(f"   总测试数: {total}")
    print(f"   通过数: {passed}")
    print(f"   失败数: {total - passed}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！简化版训练系统已准备就绪")
        print("\n🚀 使用方法:")
        print("   1. 单GPU测试: ./run_training_simple.sh 1 \"0\" test")
        print("   2. 多GPU训练: ./run_training_simple.sh 1 \"0,1,2,3\" experiment")
        print("   3. 查看文档: cat README_SIMPLE.md")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        print("   请检查相关配置和依赖")
        return 1

if __name__ == "__main__":
    exit(main())