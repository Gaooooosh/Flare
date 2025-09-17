#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LoRA的Qwen训练脚本 - 支持RoPE层去除
使用PEFT库实现高效的LoRA微调，同时支持指定层的RoPE去除
"""

import os
import sys
from pathlib import Path

# 配置wandb网络环境
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

# 早期GPU环境设置 - 必须在import torch之前
def setup_gpu_environment():
    """设置GPU环境变量 - 必须在import torch之前调用"""
    config_file = "simple_config.json"
    if Path(config_file).exists():
        # 临时导入配置管理器来读取GPU设置
        sys.path.append(str(Path(__file__).parent.parent / 'utils'))
        from config_manager import ConfigManager
        
        try:
            config_manager = ConfigManager(config_file)
            config = config_manager.get_config()
            
            if hasattr(config.environment, 'gpu_ids') and config.environment.gpu_ids:
                gpu_ids_str = ",".join(map(str, config.environment.gpu_ids))
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
                print(f"🎯 设置CUDA_VISIBLE_DEVICES: {gpu_ids_str}")
                # 验证环境变量设置
                actual_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if actual_value != gpu_ids_str:
                    print(f"⚠️ 环境变量设置异常，期望: {gpu_ids_str}, 实际: {actual_value}")
                    raise RuntimeError(f"CUDA_VISIBLE_DEVICES设置失败")
                else:
                    print(f"✅ CUDA_VISIBLE_DEVICES验证成功: {actual_value}")
            else:
                # 如果没有指定GPU，设置为使用所有可用GPU
                print("💻 未指定GPU，使用所有可用GPU")
        except Exception as e:
            print(f"⚠️ 读取GPU配置失败: {e}")
            raise
    else:
        print("📋 配置文件不存在，跳过GPU环境设置")

# 设置GPU环境
setup_gpu_environment()

# 现在可以安全导入torch
import logging
import torch
from typing import Optional, List

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'utils'))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# PEFT相关导入
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from config_manager import ConfigManager
from patch_qwen_rope import patch_qwen_rope
from simple_dataset_loader import SimpleDatasetLoader

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_lora_config():
    """获取LoRA配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,  # LoRA dropout
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # 目标模块
        bias="none",  # 不训练bias
        use_rslora=False,  # 不使用RSLoRA
    )


def verify_gradients(model):
    """验证模型参数的梯度设置（只检查，不修改）"""
    logger.info("验证模型参数梯度设置...")
    
    trainable_params = 0
    total_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_params += param.numel()
                logger.debug(f"LoRA参数: {name}")
            else:
                logger.debug(f"其他可训练参数: {name}")
    
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"LoRA参数: {lora_params:,}")
    
    if trainable_params == 0:
        raise RuntimeError("没有找到可训练的参数！LoRA配置可能有问题。")
    
    return trainable_params, total_params


def setup_model_and_tokenizer(config):
    """设置模型和分词器"""
    logger.info(f"加载模型: {config.model.model_name}")
    
    # 尝试初始化CUDA，如果失败则提供详细的错误信息
    try:
        # 强制初始化CUDA上下文
        if torch.cuda.is_available():
            torch.cuda.init()
            logger.info(f"CUDA初始化成功，可用GPU数量: {torch.cuda.device_count()}")
        else:
            raise RuntimeError("CUDA不可用")
    except Exception as e:
        logger.error(f"CUDA初始化失败: {e}")
        print("可能的解决方案:")
        print("1. 检查NVIDIA驱动程序版本是否与PyTorch兼容")
        print("2. 尝试重新安装PyTorch: pip install torch --upgrade")
        print("3. 检查CUDA_VISIBLE_DEVICES环境变量设置")
        print("4. 重启系统或重新加载NVIDIA驱动")
        raise RuntimeError("CUDA初始化失败，无法进行GPU训练")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # GPU模式设置
    device_map = "auto"
    logger.info("GPU模式：使用auto device_map")
    
    # 确定数据类型
    torch_dtype = getattr(torch, config.model.torch_dtype, torch.bfloat16)
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    if hasattr(config.model, 'use_flash_attention') and config.model.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        **model_kwargs
    )

    # 设置RoPE参数
    if hasattr(config.model, 'rope_theta') and hasattr(model.config, "rope_theta"):
        model.config.rope_theta = config.model.rope_theta
        logger.info(f"设置RoPE theta为: {config.model.rope_theta}")
    
    # 设置最大位置嵌入
    if hasattr(config.model, 'max_position_embeddings'):
        model.config.max_position_embeddings = config.model.max_position_embeddings
        logger.info(f"设置最大位置嵌入为: {config.model.max_position_embeddings}")
    
    # 保存no_rope_layers配置到模型config中
    if hasattr(config.model, 'no_rope_layers'):
        model.config.nope_layers = config.model.no_rope_layers
    
    # 应用RoPE patch（在LoRA之前）
    if hasattr(config.model, 'no_rope_layers') and config.model.no_rope_layers:
        logger.info(f"禁用RoPE层: {config.model.no_rope_layers}")
        patch_qwen_rope(model, no_rope_layers=config.model.no_rope_layers)
    
    # 获取LoRA配置并应用
    lora_config = get_lora_config()
    logger.info(f"应用LoRA配置: rank={lora_config.r}, alpha={lora_config.lora_alpha}")
    model = get_peft_model(model, lora_config)
    
    # 确保与梯度检查点兼容：需要让输入激活参与计算图，并关闭use_cache
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            logger.info("已启用输入requires_grad以兼容梯度检查点与LoRA")
    except Exception as e:
        logger.warning(f"启用输入requires_grad失败: {e}")

    # 训练时明确关闭use_cache，避免与梯度检查点冲突
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
            logger.info("已将model.config.use_cache显式设置为False")
    except Exception as e:
        logger.warning(f"设置use_cache失败: {e}")

    # 验证梯度设置（只检查，不修改）
    verify_gradients(model)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model, tokenizer


def setup_dataset(config, tokenizer):
    """设置数据集"""
    logger.info(f"加载数据集: {config.data.dataset_name}")
    
    dataset_loader = SimpleDatasetLoader(
        cache_dir=config.data.cache_dir
    )
    
    train_dataset, eval_dataset = dataset_loader.prepare_dataset(
        dataset_name=config.data.dataset_name,
        tokenizer=tokenizer,
        size_limit=config.data.dataset_size,
        validation_split=config.data.validation_split,
        max_length=config.data.max_length,
        text_column=config.data.text_column
    )
    
    return train_dataset, eval_dataset


def setup_training_args(config, output_dir):
    """设置训练参数"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=getattr(config.training, 'gradient_accumulation_steps', 1),
        learning_rate=config.training.learning_rate,
        weight_decay=getattr(config.training, 'weight_decay', 0.01),
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # 禁用wandb等
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        fp16=getattr(config.training, 'fp16', False),
        bf16=getattr(config.training, 'bf16', True),
        gradient_checkpointing=getattr(config.training, 'gradient_checkpointing', True),
        max_grad_norm=getattr(config.training, 'max_grad_norm', 1.0),
        seed=getattr(config.environment, 'seed', 42),
        data_seed=getattr(config.environment, 'seed', 42),
        remove_unused_columns=True,  # 对LoRA更安全
        # LoRA特定设置
        save_only_model=True,  # 只保存LoRA权重
    )
    
    return training_args


def run_interactive_config():
    """运行交互式配置"""
    logger.info("启动交互式配置...")
    import subprocess
    import sys
    
    # 运行交互式配置脚本
    config_script = Path(__file__).parent.parent / "interactive_config.py"
    result = subprocess.run([sys.executable, str(config_script)], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error("交互式配置失败")
        return False
    
    logger.info("交互式配置完成")
    return True


def main():
    """主训练函数 - 基于LoRA的RoPE层去除训练"""
    
    print("=" * 60)
    print("🚀 Flare LoRA + RoPE 训练系统")
    print("=" * 60)
    print()
    
    # 检查配置文件是否存在
    config_file = "simple_config.json"
    
    if not Path(config_file).exists():
        print("📋 未找到配置文件，启动交互式配置...")
        print()
        if not run_interactive_config():
            print("❌ 配置失败，退出训练")
            return
        print()
    else:
        print("📋 发现现有配置文件")
        print("   1. 使用现有配置继续训练")
        print("   2. 重新配置训练参数")
        print()
        
        while True:
            choice = input("请选择 (1/2): ").strip()
            if choice == "1":
                print("✅ 使用现有配置")
                break
            elif choice == "2":
                print("🔄 启动交互式配置...")
                print()
                if not run_interactive_config():
                    print("❌ 配置失败，退出训练")
                    return
                break
            else:
                print("❌ 无效选择，请输入 1 或 2")
    
    print()
    print("=" * 60)
    print("🔧 开始LoRA训练准备...")
    print("=" * 60)
    
    # 加载配置
    if not Path(config_file).exists():
        logger.error(f"配置文件不存在: {config_file}")
        print("❌ 配置文件丢失，请重新运行程序")
        return
    
    config_manager = ConfigManager(config_file)
    config = config_manager.get_config()
    
    # GPU环境已在文件开头设置
    
    # 设置输出目录 - 强制使用/work/xiaoyonggao作为根目录
    base_work_dir = Path("/work/xiaoyonggao")
    if config.output.experiment_name:
        output_dir = base_work_dir / f"{config.output.experiment_name}_lora"
    else:
        output_dir = base_work_dir / "flare_lora_training"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 设置模型和分词器
    print("📦 加载模型和分词器（应用LoRA）...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 设置数据集
    print("📊 准备数据集...")
    train_dataset, eval_dataset = setup_dataset(config, tokenizer)
    
    # 设置训练参数
    print("⚙️ 配置训练参数...")
    training_args = setup_training_args(config, str(output_dir))
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 创建训练器
    print("🏗️ 创建LoRA训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print()
    print("=" * 60)
    print("🎯 开始LoRA训练...")
    print("=" * 60)
    
    # 开始训练
    trainer.train()
    
    # 保存LoRA模型
    final_model_dir = output_dir / "final_lora_model"
    print(f"💾 保存LoRA模型到: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # 保存完整的配置信息
    config_save_path = final_model_dir / "training_config.json"
    config_manager.save_config(str(config_save_path))
    
    print()
    print("=" * 60)
    print("🎉 LoRA训练完成！")
    print(f"📁 LoRA模型保存位置: {final_model_dir}")
    print(f"📋 训练配置保存位置: {config_save_path}")
    print("=" * 60)
    print()
    print("💡 使用说明:")
    print("1. LoRA权重已保存，可以与原始模型合并使用")
    print("2. 指定的RoPE层已被禁用")
    print("3. 可以使用PEFT库加载LoRA权重进行推理")


if __name__ == "__main__":
    main()