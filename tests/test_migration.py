#!/usr/bin/env python3
"""
迁移测试脚本
用于验证新训练系统的各项功能是否正常工作
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class MigrationTester:
    """迁移测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.base_dir = Path("/home/xiaoyonggao/Flare")
        self.work_dir = Path("/work/xiaoyonggao")
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        logger.info(f"运行测试: {test_name}")
        try:
            result = test_func()
            self.test_results[test_name] = {"status": "PASS", "result": result}
            logger.info(f"✅ {test_name}: PASS")
        except Exception as e:
            self.test_results[test_name] = {"status": "FAIL", "error": str(e)}
            logger.error(f"❌ {test_name}: FAIL - {e}")
    
    def test_environment(self) -> Dict[str, Any]:
        """测试环境配置"""
        results = {}
        
        # Python版本
        results["python_version"] = sys.version
        
        # PyTorch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["cuda_version"] = torch.version.cuda
            results["gpu_count"] = torch.cuda.device_count()
            results["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        # 检查关键包
        packages = ["transformers", "datasets", "accelerate", "tensorboard"]
        for pkg in packages:
            try:
                module = __import__(pkg)
                results[f"{pkg}_version"] = getattr(module, "__version__", "unknown")
            except ImportError:
                results[f"{pkg}_version"] = "NOT_INSTALLED"
        
        return results
    
    def test_file_structure(self) -> Dict[str, bool]:
        """测试文件结构"""
        required_files = [
            "train_qwen_multi_gpu.py",
            "evaluate_model_enhanced.py",
            "training_config.json",
            "run_training.sh",
            "patch_qwen_rope.py",
            "README_MIGRATION.md",
            "requirements_migration.txt"
        ]
        
        results = {}
        for file_name in required_files:
            file_path = self.base_dir / file_name
            results[file_name] = file_path.exists()
        
        return results
    
    def test_script_syntax(self) -> Dict[str, bool]:
        """测试脚本语法"""
        python_files = [
            "train_qwen_multi_gpu.py",
            "evaluate_model_enhanced.py",
            "patch_qwen_rope.py"
        ]
        
        results = {}
        for file_name in python_files:
            file_path = self.base_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    compile(code, str(file_path), 'exec')
                    results[file_name] = True
                except SyntaxError as e:
                    results[file_name] = f"SyntaxError: {e}"
                except Exception as e:
                    results[file_name] = f"Error: {e}"
            else:
                results[file_name] = "FILE_NOT_FOUND"
        
        return results
    
    def test_config_file(self) -> Dict[str, Any]:
        """测试配置文件"""
        config_file = self.base_dir / "training_config.json"
        
        if not config_file.exists():
            raise FileNotFoundError("配置文件不存在")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要的配置项
        required_sections = ["model_args", "data_args", "training_args"]
        results = {"sections": {}}
        
        for section in required_sections:
            results["sections"][section] = section in config
        
        results["config_valid"] = all(results["sections"].values())
        return results
    
    def test_work_directory(self) -> Dict[str, Any]:
        """测试工作目录"""
        results = {}
        
        # 检查目录是否存在
        results["exists"] = self.work_dir.exists()
        
        if not self.work_dir.exists():
            # 尝试创建目录
            try:
                self.work_dir.mkdir(parents=True, exist_ok=True)
                results["created"] = True
            except Exception as e:
                results["created"] = False
                results["create_error"] = str(e)
        
        # 检查写入权限
        if self.work_dir.exists():
            test_file = self.work_dir / "test_write.txt"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # 删除测试文件
                results["writable"] = True
            except Exception as e:
                results["writable"] = False
                results["write_error"] = str(e)
        
        return results
    
    def test_gpu_manager(self) -> Dict[str, Any]:
        """测试GPU管理器"""
        try:
            # 导入GPU管理器
            sys.path.insert(0, str(self.base_dir))
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
            from train_qwen_multi_gpu import GPUManager
            
            gpu_info = GPUManager.get_available_gpus()
            
            results = {
                "gpu_info": gpu_info,
                "has_gpus": len(gpu_info["available"]) > 0
            }
            
            if results["has_gpus"]:
                # 测试GPU选择
                selected = GPUManager.select_gpus()
                results["gpu_selection"] = selected
            
            return results
            
        except Exception as e:
            raise Exception(f"GPU管理器测试失败: {e}")
    
    def test_model_loading(self) -> Dict[str, Any]:
        """测试模型加载（轻量级测试）"""
        try:
            from transformers import AutoTokenizer
            
            # 测试分词器加载
            model_name = "Qwen/Qwen2.5-0.5B"  # 使用小模型进行测试
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # 测试分词
            test_text = "Hello, world!"
            tokens = tokenizer(test_text, return_tensors="pt")
            
            results = {
                "tokenizer_loaded": True,
                "vocab_size": tokenizer.vocab_size,
                "test_tokens": tokens["input_ids"].shape[1]
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"模型加载测试失败: {e}")
    
    def test_patch_qwen_rope(self) -> Dict[str, Any]:
        """测试RoPE修改功能"""
        try:
            sys.path.insert(0, str(self.base_dir))
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
            from patch_qwen_rope import patch_qwen_rope
            
            # 检查函数是否可调用
            results = {
                "function_exists": callable(patch_qwen_rope),
                "function_signature": str(patch_qwen_rope.__doc__)
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"RoPE修改测试失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行迁移测试...")
        
        # 定义测试列表
        tests = [
            ("环境配置", self.test_environment),
            ("文件结构", self.test_file_structure),
            ("脚本语法", self.test_script_syntax),
            ("配置文件", self.test_config_file),
            ("工作目录", self.test_work_directory),
            ("GPU管理器", self.test_gpu_manager),
            ("模型加载", self.test_model_loading),
            ("RoPE修改", self.test_patch_qwen_rope),
        ]
        
        # 运行测试
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
    
    def generate_report(self):
        """生成测试报告"""
        logger.info("\n" + "="*50)
        logger.info("迁移测试报告")
        logger.info("="*50)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = result["status"]
            if status == "PASS":
                passed += 1
                logger.info(f"✅ {test_name}: 通过")
            else:
                failed += 1
                logger.error(f"❌ {test_name}: 失败 - {result.get('error', '未知错误')}")
        
        logger.info(f"\n总计: {passed + failed} 个测试")
        logger.info(f"通过: {passed}")
        logger.info(f"失败: {failed}")
        
        if failed == 0:
            logger.info("\n🎉 所有测试通过！迁移系统准备就绪。")
        else:
            logger.warning(f"\n⚠️  有 {failed} 个测试失败，请检查相关配置。")
        
        # 保存详细报告
        report_file = self.base_dir / "migration_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n详细报告已保存到: {report_file}")
        
        # 返回测试是否全部通过
        return failed == 0


def main():
    """主函数"""
    print("🚀 Qwen训练系统迁移测试")
    print("="*50)
    
    tester = MigrationTester()
    tester.run_all_tests()
    success = tester.generate_report()
    
    if success:
        print("\n✅ 迁移测试完成，系统准备就绪！")
        print("\n下一步操作:")
        print("1. 安装依赖: pip install -r requirements_migration.txt")
        print("2. 阅读文档: cat README_MIGRATION.md")
        print("3. 开始训练: ./run_training.sh 1 auto test_experiment")
        return 0
    else:
        print("\n❌ 迁移测试发现问题，请检查失败的测试项。")
        return 1


if __name__ == "__main__":
    exit(main())