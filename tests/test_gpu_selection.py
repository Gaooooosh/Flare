#!/usr/bin/env python3
"""
GPU选择功能测试脚本
用于验证训练脚本的GPU选择功能是否正常工作
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, timeout=30):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "命令超时"
    except Exception as e:
        return -1, "", str(e)


def test_gpu_selection():
    """测试GPU选择功能"""
    print("🧪 测试GPU选择功能")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        {
            "name": "自动选择GPU",
            "cmd": "python train_qwen_multi_gpu.py --help | grep -A 5 'gpu_ids\|gpu_type'",
            "description": "检查GPU参数是否存在"
        },
        {
            "name": "GPU管理器导入测试",
            "cmd": "python -c 'import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"training\")); from train_qwen_multi_gpu import GPUManager; print(\"GPU管理器导入成功\")'",
            "description": "测试GPU管理器是否可以正常导入"
        },
        {
            "name": "GPU信息获取测试",
            "cmd": "python -c 'import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"training\")); from train_qwen_multi_gpu import GPUManager; info = GPUManager.get_available_gpus(); print(f\"可用GPU: {info[\\\"available\\\"]}\")'",
            "description": "测试获取GPU信息功能"
        },
        {
            "name": "A800 GPU选择测试",
            "cmd": "python -c 'import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"training\")); from train_qwen_multi_gpu import GPUManager; gpus = GPUManager.select_gpus(gpu_type=\"A800\"); print(f\"A800 GPU: {gpus}\")'",
            "description": "测试A800 GPU选择"
        },
        {
            "name": "A40 GPU选择测试",
            "cmd": "python -c 'import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"training\")); from train_qwen_multi_gpu import GPUManager; gpus = GPUManager.select_gpus(gpu_type=\"A40\"); print(f\"A40 GPU: {gpus}\")'",
            "description": "测试A40 GPU选择"
        },
        {
            "name": "手动GPU选择测试",
            "cmd": "python -c 'import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"training\")); from train_qwen_multi_gpu import GPUManager; gpus = GPUManager.select_gpus(gpu_ids=[0, 1]); print(f\"手动选择GPU: {gpus}\")'",
            "description": "测试手动GPU选择"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   描述: {test['description']}")
        print(f"   命令: {test['cmd'][:80]}{'...' if len(test['cmd']) > 80 else ''}")
        
        returncode, stdout, stderr = run_command(test['cmd'])
        
        if returncode == 0:
            print(f"   ✅ 成功")
            if stdout.strip():
                print(f"   输出: {stdout.strip()}")
            results.append((test['name'], True, stdout.strip()))
        else:
            print(f"   ❌ 失败")
            if stderr.strip():
                print(f"   错误: {stderr.strip()}")
            results.append((test['name'], False, stderr.strip()))
    
    return results


def test_startup_script():
    """测试启动脚本的GPU参数"""
    print("\n\n🚀 测试启动脚本GPU参数")
    print("=" * 50)
    
    # 检查启动脚本是否存在
    script_path = Path("run_training.sh")
    if not script_path.exists():
        print("❌ 启动脚本不存在")
        return [("启动脚本存在性", False, "文件不存在")]
    
    # 检查脚本内容
    with open(script_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("GPU_IDS参数", "GPU_IDS" in content),
        ("gpu_ids参数处理", "--gpu_ids" in content),
        ("manual GPU类型", "manual" in content),
        ("GPU列表转换", "GPU_LIST" in content)
    ]
    
    results = []
    for check_name, passed in checks:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {check_name}: {status}")
        results.append((check_name, passed, ""))
    
    return results


def test_config_examples():
    """测试配置示例文件"""
    print("\n\n📋 测试配置示例文件")
    print("=" * 50)
    
    config_file = Path("training_config_gpu_examples.json")
    
    if not config_file.exists():
        print("❌ 配置示例文件不存在")
        return [("配置示例文件", False, "文件不存在")]
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("✅ 配置文件格式正确")
        
        # 检查关键示例
        examples = [
            "example_1_auto_selection",
            "example_3_manual_gpu_selection",
            "gpu_mapping_reference"
        ]
        
        results = []
        for example in examples:
            if example in config:
                print(f"   ✅ {example}: 存在")
                results.append((example, True, ""))
            else:
                print(f"   ❌ {example}: 缺失")
                results.append((example, False, "示例缺失"))
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return [("配置文件格式", False, str(e))]
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return [("配置文件读取", False, str(e))]


def generate_report(all_results):
    """生成测试报告"""
    print("\n\n📊 测试报告")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n📂 {category}:")
        for test_name, passed, details in results:
            total_tests += 1
            if passed:
                passed_tests += 1
                print(f"   ✅ {test_name}")
            else:
                print(f"   ❌ {test_name}: {details}")
    
    print(f"\n📈 总结:")
    print(f"   总测试数: {total_tests}")
    print(f"   通过数: {passed_tests}")
    print(f"   失败数: {total_tests - passed_tests}")
    print(f"   成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有GPU选择功能测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
        return False


def main():
    """主函数"""
    print("🔧 GPU选择功能测试")
    print("=" * 60)
    
    # 运行各项测试
    all_results = {}
    
    # 测试GPU选择功能
    all_results["GPU选择功能"] = test_gpu_selection()
    
    # 测试启动脚本
    all_results["启动脚本"] = test_startup_script()
    
    # 测试配置示例
    all_results["配置示例"] = test_config_examples()
    
    # 生成报告
    success = generate_report(all_results)
    
    if success:
        print("\n✅ GPU选择功能已完全实现并测试通过！")
        print("\n🚀 使用方法:")
        print("   1. 查看GPU信息: python check_gpus.py")
        print("   2. 手动选择GPU: ./run_training.sh 1 manual exp \"0,1,2,3\"")
        print("   3. 按类型选择: ./run_training.sh 1 A800 exp")
        print("   4. 查看配置示例: cat training_config_gpu_examples.json")
        return 0
    else:
        print("\n❌ 部分功能测试失败，请检查相关实现")
        return 1


if __name__ == "__main__":
    exit(main())