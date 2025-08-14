# 项目结构说明

本项目已按照功能模块重新整理，目录结构如下：

## 📁 目录结构

```
Flare/
├── configs/                    # 配置文件
│   ├── training_config.json           # 原始训练配置
│   ├── training_config_simple.json    # 简化版训练配置
│   └── training_config_gpu_examples.json  # GPU配置示例
│
├── data/                       # 数据文件目录
│   └── (数据文件存放位置)
│
├── docs/                       # 文档
│   ├── README_MIGRATION.md            # 迁移指南
│   └── README_SIMPLE.md               # 简化版使用指南
│
├── results/                    # 训练和评估结果
│   ├── attention_scores-*.csv         # 注意力分数数据
│   ├── attention_scores_comparison*.png # 注意力对比图
│   ├── model_responses-*.jsonl        # 模型响应数据
│   └── migration_test_report.json     # 迁移测试报告
│
├── scripts/                    # 启动脚本
│   ├── run_training.sh                # 原始训练启动脚本
│   └── run_training_simple.sh         # 简化版训练启动脚本
│
├── tests/                      # 测试脚本
│   ├── check_gpus.py                  # GPU检查工具
│   ├── test_gpu_selection.py          # GPU选择测试
│   ├── test_migration.py              # 迁移测试
│   └── test_simple_setup.py           # 简化版设置测试
│
├── training/                   # 训练脚本
│   ├── train_qwen_multi_gpu.py        # 原始多GPU训练脚本
│   └── train_qwen_simple.py           # 简化版训练脚本
│
├── utils/                      # 工具和评估脚本
│   ├── evaluate_model.py              # 模型评估
│   ├── evaluate_model_enhanced.py     # 增强版模型评估
│   ├── eval_wikitext_ppl.py           # WikiText困惑度评估
│   ├── mmlu_evaluate.py               # MMLU评估
│   ├── niah_probe.py                  # NIAH探测
│   ├── patch_qwen_rope.py             # Qwen RoPE补丁
│   ├── summarize_results.py           # 结果汇总
│   ├── untils.py                      # 工具函数
│   └── visualize_attention.py         # 注意力可视化
│
├── pyproject.toml              # 项目配置
├── requirements.txt            # Python依赖
├── requirements_migration.txt  # 迁移依赖
└── uv.lock                     # 依赖锁定文件
```

## 🚀 快速开始

### 1. 简化版训练（推荐）
```bash
# 使用启动脚本
./scripts/run_training_simple.sh --stage 1 --gpu_ids "0,1" --exp_name "my_experiment"

# 直接使用Python脚本
cd training
python train_qwen_simple.py --config ../configs/training_config_simple.json --gpu_ids "0,1"
```

### 2. 检查GPU状态
```bash
cd tests
python check_gpus.py
```

### 3. 运行测试
```bash
cd tests
python test_simple_setup.py
```

## 📋 主要功能

- **训练脚本**: 支持原始版本和简化版本
- **GPU管理**: 自动检测和选择GPU
- **配置管理**: 灵活的JSON配置文件
- **评估工具**: 多种模型评估方法
- **测试套件**: 完整的功能测试
- **文档**: 详细的使用指南

## 📝 注意事项

1. 所有脚本都需要在项目根目录下运行
2. 确保GPU环境正确配置
3. 查看 `docs/` 目录下的详细文档
4. 测试脚本可以验证环境配置是否正确