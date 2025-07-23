"""
单卡A40全量微调Qwen2.5-3B
支持：
1. Flash-Attention 2
2. 长上下文（通过rope_scaling）
3. 指定层禁用RoPE
4. 本地RedPajama jsonl
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import json
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from patch_qwen_rope import patch_qwen_rope
from transformers import TrainerCallback

# 添加冻结/解冻函数
def set_freeze_layers(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze
    # 解冻指定的nope_layers层
    if model.config.nope_layers:
        for layer_idx in model.config.nope_layers:
            # 修正层访问路径并添加调试信息
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
            elif hasattr(model, 'layers'):
                layers = model.layers
            else:
                raise AttributeError("Model has no layers attribute")
            
            if 0 <= layer_idx < len(layers):
                print(f"解冻层 {layer_idx}")
                for param in layers[layer_idx].parameters():
                    param.requires_grad = True
                    print(f"  参数 {param.name}: requires_grad={param.requires_grad}")
            else:
                print(f"层索引 {layer_idx} 超出范围，总层数 {len(layers)}")
    return model

# 添加早停回调
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_loss = metrics.get("eval_loss", float("inf"))
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
        return control

# ----------- 1. 自定义参数 -----------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-3B")
    rope_theta: float = field(default=1000000.0)  # 长上下文基频
    max_position_embeddings: int = field(default=32768)  # 目标长度
    no_rope_layers: List[int] = field(default_factory=list)  # 禁用RoPE的层号（0-based）
    stage1_train_data_path: str = field(default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0000/en_large.json.gz")  # 阶段一训练数据
    stage2_train_data_path: str = field(default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0000/en_task.json.gz")  # 阶段二训练数据
    stage1_dataset_size: int = field(default=100000)  # 阶段一数据集大小
    stage2_dataset_size: int = field(default=10000)   # 阶段二数据集大小
    stage1_max_seq_len: int = field(default=8192)     # 阶段一最大序列长度
    stage2_max_seq_len: int = field(default=32768)    # 阶段二最大序列长度

@dataclass
class DataArguments:
    data_path: str = field(default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/{0000}/en_*.json.gz")
    max_seq_len: int = field(default=32768)

# ----------- 3. 处理数据 -----------
def make_data_module(data_args, tokenizer, stage=1):
    import os
    from datasets import Dataset

    # 检查是否存在已保存的处理后数据集
    # 根据阶段选择不同的数据集路径和大小
    if stage == 1:
        data_path = data_args.stage1_train_data_path
        max_seq_len = data_args.stage1_max_seq_len
        dataset_size = data_args.stage1_dataset_size
        cache_dir = "./stage1_train_dataset"
    else:
        data_path = data_args.stage2_train_data_path
        max_seq_len = data_args.stage2_max_seq_len
        dataset_size = data_args.stage2_dataset_size
        cache_dir = "./stage2_train_dataset"

    # 检查是否存在已保存的处理后数据集
    if os.path.exists(f"{cache_dir}_train") and os.path.exists(f"{cache_dir}_eval"):
        mixed_train_dataset = Dataset.load_from_disk(f"{cache_dir}_train")
        mixed_eval_dataset = Dataset.load_from_disk(f"{cache_dir}_eval")
        print(f"Loaded existing processed dataset from {cache_dir}")
    else:
        data_files = data_path
        print(f"Loading data from: {data_files}")
        raw_dataset = load_dataset("json", data_files=data_files, split="train")
        actual_size = len(raw_dataset)
        selected_size = min(dataset_size, actual_size)
        # raw_dataset_dict = raw_dataset.select(range(selected_size))
        raw_dataset_dict = raw_dataset.select(range(10000))
        # 划分训练集和验证集 (90%训练, 10%验证)
        raw_dataset_split = raw_dataset_dict.train_test_split(test_size=0.1, seed=42)
        raw_train_dataset = raw_dataset_split["train"]
        raw_eval_dataset = raw_dataset_split["test"]
        # 限制评估数据集大小以减少内存占用
        eval_size_limit = 100
        raw_eval_dataset = raw_eval_dataset.select(range(min(len(raw_eval_dataset), eval_size_limit)))

        def tokenize(example, max_length):
            tokens = tokenizer(
                example.get("raw_content", example.get("content", example.get("document", ""))),
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_overflowing_tokens=False,
            )
            return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}


        # 按比例混合两种上下文长度
        import random
        random.seed(42)  # 设置随机种子
        # 原训练集处理代码
        mixed_train_dataset = []
        # 评估阶段使用更短的序列长度减少内存占用
        lengths = [max_seq_len]
        proportions = [1.0]
        for length, proportion in zip(lengths, proportions):
            sampled_dataset = raw_train_dataset.shuffle(seed=42).select(range(int(len(raw_train_dataset) * proportion)))
            sampled_dataset = sampled_dataset.map(lambda x: tokenize(x, length), num_proc=16).filter(lambda x: len(x["input_ids"]) > 0)
            mixed_train_dataset.extend(sampled_dataset)
        mixed_train_dataset = Dataset.from_list(mixed_train_dataset)
        
        # 添加验证集处理
        mixed_eval_dataset = []
        eval_max_seq_len = min(max_seq_len, 256)
        for length, proportion in zip(lengths, proportions):
            sampled_dataset = raw_eval_dataset.shuffle(seed=42).select(range(int(len(raw_eval_dataset) * proportion)))
            sampled_dataset = sampled_dataset.map(lambda x: tokenize(x, eval_max_seq_len), num_proc=16).filter(lambda x: len(x["input_ids"]) > 0)
            mixed_eval_dataset.extend(sampled_dataset)
        mixed_eval_dataset = Dataset.from_list(mixed_eval_dataset)
        mixed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        mixed_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        mixed_train_dataset.save_to_disk(f"{cache_dir}_train")
        mixed_eval_dataset.save_to_disk(f"{cache_dir}_eval")
        print(f"Saved processed dataset to {cache_dir}")
        
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return dict(
        train_dataset=mixed_train_dataset,
        eval_dataset=mixed_eval_dataset,
        data_collator=data_collator,
    )
# ----------- 4. 主函数 -----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/raid_sdh/home/xyg/PRETRAINED_MODEL/qwen-3B")
    parser.add_argument("--stage1_train_data_path", type=str, default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0000/en_middle.json.gz")
    parser.add_argument("--stage2_train_data_path", type=str, default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0000/en_head.json.gz")
    parser.add_argument("--output_dir", type=str, default="./output_qwen3b_redpajama_allrope")
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--stage", type=int, help="Training stage (1 or 2)",default=2)
    parser.add_argument("--no_rope_layers", type=int, nargs="*", default=[])#list(range(20,34))
    parser.add_argument("--stage1_dataset_size", type=int, default=100000)
    parser.add_argument("--stage2_dataset_size", type=int, default=10000)
    parser.add_argument("--stage1_max_seq_len", type=int, default=4096)
    parser.add_argument("--stage2_max_seq_len", type=int, default=4096)
    args = parser.parse_args()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 模型
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map='balanced',
        torch_dtype=torch.bfloat16,
    )
    # patch_qwen_rope(model, no_rope_layers=args.no_rope_layers)
    # 长上下文：调整RoPE基频
    if hasattr(model.config, "rope_theta"):
        model.config.rope_theta = args.rope_theta
    # 根据评估阶段动态调整位置嵌入大小
    eval_max_seq_len = min(args.stage2_max_seq_len, 256)
    model.config.max_position_embeddings = args.stage1_max_seq_len
    model.config.nope_layers = args.no_rope_layers

    # 添加评估前清理CUDA缓存的回调
    from transformers import TrainerCallback
    import torch

    class CudaCacheClearCallback(TrainerCallback):
        def on_evaluate(self, args, state, control,** kwargs):
            torch.cuda.empty_cache()

    # ------------------- 阶段二：解冻全模型，整体微调 -------------------
    print("\n===== 阶段二：解冻全模型，整体微调 ======")
    # 解冻所有层
    model = set_freeze_layers(model, freeze=False)
        # 调试：打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
    # 添加早停回调
    early_stopping_callback = EarlyStoppingCallback(patience=5)
    # 阶段二数据
    data_module_stage2 = make_data_module(args, tokenizer, stage=2)

    # 阶段二训练参数
    training_args_stage2 = TrainingArguments(
        output_dir=f"{args.output_dir}/stage2",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        eval_steps=20,
        eval_on_start=True,
        eval_strategy="steps",  
        learning_rate=3e-5,  # 阶段二：非常低的学习率
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        dataloader_num_workers=4,
        max_steps=2000,  # 阶段二：较多步数
        gradient_checkpointing=True,
        report_to="tensorboard",
        resume_from_checkpoint=False, 
    )

    trainer_stage2 = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args_stage2,
        callbacks=[early_stopping_callback, CudaCacheClearCallback()],
        **data_module_stage2,
    )

    trainer_stage2.train()
    trainer_stage2.save_model(args.output_dir)


if __name__ == "__main__":
    main()