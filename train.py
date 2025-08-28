#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flare 交互式训练系统 - 唯一启动入口
"""

import sys
from pathlib import Path

def main():
    """启动训练系统"""
    print("🚀 启动 Flare 交互式训练系统...")
    print()
    
    # 导入并运行训练脚本
    training_script = Path(__file__).parent / "training" / "train_simple.py"
    
    if not training_script.exists():
        print("❌ 训练脚本不存在")
        return 1
    
    # 执行训练脚本
    import subprocess
    result = subprocess.run([sys.executable, str(training_script)])
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())