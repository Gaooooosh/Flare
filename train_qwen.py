"""
单卡A40全量微调Qwen2.5-3B
支持：
1. Flash-Attention 2
2. 长上下文（通过rope_scaling）
3. 指定层禁用RoPE
4. 本地RedPajama jsonl
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from patch_qwen_rope import patch_qwen_rope
from eval_wikitext_ppl import WikiPPLCallback
from transformers import TrainerCallback

class PPLCallback(TrainerCallback):
    def __init__(self, tokenizer, max_seq_len, eval_every_steps=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_steps == 0 and state.global_step > 0:
            ppl = eval_ppl(model, self.tokenizer, self.max_seq_len)
            print(f"Step {state.global_step}: WikiText PPL = {ppl:.2f}")
            # 写 TensorBoard
            if args.report_to == "tensorboard":
                state.tb_writer.add_scalar("eval/wikitext_ppl", ppl, state.global_step)
# ----------- 1. 自定义参数 -----------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-3B")
    rope_theta: float = field(default=1000000.0)  # 长上下文基频
    max_position_embeddings: int = field(default=32768)  # 目标长度
    no_rope_layers: List[int] = field(default_factory=list)  # 禁用RoPE的层号（0-based）

@dataclass
class DataArguments:
    data_path: str = field(default="data/redpajama_sample.jsonl")
    max_seq_len: int = field(default=32768)

# ----------- 3. 处理数据 -----------
def make_data_module(data_args, tokenizer):
    raw_dataset_dict = load_dataset(data_args.data_path, split="train")  # ← 直接加载
    train_dataset = raw_dataset_dict.select(range(500000))

    def tokenize(example, max_length):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    # 确保数据集列名与模型输入匹配
    train_dataset = train_dataset.map(tokenize, fn_kwargs={"max_length": data_args.max_seq_len}, num_proc=16)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # 按比例混合两种上下文长度
    import random
    random.seed(42)  # 设置随机种子
    mixed_train_dataset = []
    lengths = [8192, data_args.max_seq_len]  # 示例比例
    proportions = [0.25, 0.75]  # 示例比例
    for length, proportion in zip(lengths, proportions):
        sampled_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * proportion)))
        sampled_dataset = sampled_dataset.map(lambda x: tokenize(x, length), num_proc=16)
        mixed_train_dataset.extend(sampled_dataset)

    # 将混合数据集转换为Dataset对象
    from datasets import Dataset
    mixed_train_dataset = Dataset.from_list(mixed_train_dataset)
    mixed_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
# ----------- 4. 主函数 -----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/raid_sdh/home/xyg/PRETRAINED_MODEL/qwen-3B")
    parser.add_argument("--data_path", type=str, default="/raid_sdh/home/xyg/RedPajama")
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default="./output_qwen3b_redpajama_allrope")
    parser.add_argument("--rope_theta", type=float, default=1000000.0)
    parser.add_argument("--no_rope_layers", type=int, nargs="*", default=[])#list(range(20,34))
    args = parser.parse_args()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map='auto'
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,  # 1×32=32 samples/step
        learning_rate=1e-5,
        warmup_steps=100,
        bf16=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=4,
        max_steps=300,
        gradient_checkpointing=True,
        # eval_steps=1,
        # deepspeed="ds_config_zero1.json",  # 单卡可用ZeRO-1
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
        # callbacks=[WikiPPLCallback(tokenizer, max_len=args.max_seq_len, eval_every=training_args.eval_steps)]
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()