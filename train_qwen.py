"""
单卡A40全量微调Qwen2.5-3B
支持：
1. Flash-Attention 2
2. 长上下文（通过rope_scaling）
3. 指定层禁用RoPE
4. 本地RedPajama jsonl
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
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

# ----------- 1. 自定义参数 -----------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-3B")
    rope_theta: float = field(default=1000000.0)  # 长上下文基频
    max_position_embeddings: int = field(default=32768)  # 目标长度
    no_rope_layers: List[int] = field(default_factory=list)  # 禁用RoPE的层号（0-based）

@dataclass
class DataArguments:
    data_path: str = field(default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/{0000}/en_*.json.gz")
    max_seq_len: int = field(default=32768)

# ----------- 3. 处理数据 -----------
def make_data_module(data_args, tokenizer):
    import os
    from datasets import Dataset

    # 检查是否存在已保存的处理后数据集
    if os.path.exists("./mixed_train_dataset"):
        mixed_train_dataset = Dataset.load_from_disk("./mixed_train_dataset")
        print("Loaded existing processed dataset from ./mixed_train_dataset")
    else:
        data_files = data_args.data_path
        print(f"Loading data from: {data_files}")
        raw_dataset_dict = load_dataset("json", data_files=data_files, split="train")  # 递归加载所有压缩JSON文件(依赖扩展名自动解压)
        train_dataset = raw_dataset_dict.select(range(1000))

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
        mixed_train_dataset = []
        lengths = [data_args.max_seq_len]  # 固定长度
        proportions = [1.0]  # 100%比例
        for length, proportion in zip(lengths, proportions):
            sampled_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * proportion)))
            sampled_dataset = sampled_dataset.map(lambda x: tokenize(x, length), num_proc=16).filter(lambda x: len(x["input_ids"]) > 0)

            mixed_train_dataset.extend(sampled_dataset)

        # 将混合数据集转换为Dataset对象
        mixed_train_dataset = Dataset.from_list(mixed_train_dataset)
        mixed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        mixed_train_dataset.save_to_disk("./mixed_train_dataset")
        print("Saved processed dataset to ./mixed_train_dataset")
        
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return dict(
        train_dataset=mixed_train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
# ----------- 4. 主函数 -----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/raid_sdh/home/xyg/PRETRAINED_MODEL/qwen-3B")
    parser.add_argument("--data_path", type=str, default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0000/en_middle.json.gz")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="./output_qwen3b_redpajama_nope20~4k")
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--no_rope_layers", type=int, nargs="*", default=list(range(20,34)))#list(range(20,34))
    args = parser.parse_args()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    patch_qwen_rope(model, no_rope_layers=args.no_rope_layers)
    # 长上下文：调整RoPE基频
    if hasattr(model.config, "rope_theta"):
        model.config.rope_theta = args.rope_theta
    model.config.max_position_embeddings = args.max_seq_len
    model.config.nope_layers = args.no_rope_layers
    # 数据
    data_module = make_data_module(args, tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # 1×32=32 samples/step
        learning_rate=3e-5,
        warmup_steps=100,
        bf16=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=4,
        max_steps=500,
        report_to="tensorboard",

        fsdp="full_shard auto_wrap",
        fsdp_config={
            "min_num_params": 100000000,
            "wrap_module": "QwenBlock",
            "use_orig_params": True,
            "activation_checkpointing": True,
        },
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()