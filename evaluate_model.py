"""
使用niah探针实验评估模型
支持：
1. 配置测试的上下文长度和探针的深度
2. 汇报所有层的注意力模式
3. 分层、分头保存注意力分数

作者：xyg
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
import pandas as pd
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from niah_probe import NIAHProbe
from patch_qwen_rope import patch_qwen_rope
from datasets import load_dataset
from tqdm import tqdm
# 定义评估参数的数据类
@dataclass
class EvaluationArguments:
    model_name_or_path: str = field(default="/raid_sdh/home/xyg/output_qwen3b_redpajama_nope-20~")
    data_path: str = field(default="/raid_sdh/home/xyg/RedPajama")
    max_seq_len: int = field(default=32768)
    probe_depth: int = field(default=10)

# 定义评估模型的函数
def evaluate_model(eval_args):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        eval_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
          output_attentions=True,
        device_map='auto'
    )

    # 读取训练好的模型的配置
    config_path = os.path.join(eval_args.model_name_or_path, 'config.json')
    print(f'Reading config from {config_path}')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f'Config file not found: {config_path}')
        return
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON from config file: {e}')
        return

    # 应用猴子补丁
    print(f'Applying monkey patch with no_rope_layers: {config.get("nope_layers", [])}')
    try:
        patch_qwen_rope(model, no_rope_layers=config.get('nope_layers', []))
    except Exception as e:
        print(f'Error applying monkey patch: {e}')
        return

    # 初始化探针
    probe = NIAHProbe(model, tokenizer, max_seq_len=eval_args.max_seq_len, probe_depth=eval_args.probe_depth)

    # 加载数据集
    raw_dataset_dict = load_dataset(eval_args.data_path, split="train")
    eval_dataset = raw_dataset_dict.select(range(5000))

    # 定义tokenize函数
    def tokenize(example, max_length):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    # 对数据集进行map操作
    eval_dataset = eval_dataset.map(tokenize, fn_kwargs={"max_length": eval_args.max_seq_len}, num_proc=4)
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # 评估注意力模式
    sink_scores = {}
    probe_scores = {}
    other_scores = {}
    
    for layer in range(model.config.num_hidden_layers):
        sink_score = []
        probe_score = []
        other_score = []
        
        for batch in tqdm(eval_dataset):
            input_ids = batch['input_ids'].unsqueeze(0).to(model.device)
            attention_scores = probe.get_attention_scores(input_ids)
            layer_data = attention_scores.get(layer, {})
            # 从探针结果中获取当前层的分数
            
            # 计算sink token注意力得分
            if 'sink' in layer_data:
                sink_score.append(sum(layer_data['sink']) / len(layer_data['sink']))
            
            # 计算探针注意力得分
            if eval_args.probe_depth > 0 and 'probe' in layer_data:
                probe_score.append(sum(layer_data['probe']) / len(layer_data['probe']))
            
            # 计算其他token注意力得分
            if 'other' in layer_data:
                other_score.append(sum(layer_data['other']) / len(layer_data['other']) if layer_data['other'] else 0)
        
        sink_scores[layer] = sum(sink_score) / len(sink_score) if sink_score else 0
        if eval_args.probe_depth > 0:
            probe_scores[layer] = sum(probe_score) / len(probe_score) if probe_score else 0
        other_scores[layer] = sum(other_score) / len(other_score) if other_score else 0

    # 汇报结果
    for layer in range(model.config.num_hidden_layers):
        avg_sink = sink_scores.get(layer, 0)
        
        if eval_args.probe_depth > 0:
            avg_probe = probe_scores.get(layer, 0)
            avg_other = other_scores.get(layer, 0)
            print(f"Layer {layer}: sink={avg_sink:.4f}, probe={avg_probe:.4f}, other={avg_other:.4f}")
        else:
            print(f"Layer {layer}: sink={avg_sink:.4f}")

    # 保存注意力分数
    results = []
    for layer in sink_scores:
        result = {
            'Layer': layer,
            'Sink_Score': sink_scores[layer],
            'Other_Score': other_scores[layer]
        }
        if eval_args.probe_depth > 0:
            result['Probe_Score'] = probe_scores[layer]
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv("attention_scores.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/raid_sdh/home/xyg/output_qwen3b_redpajama_allrope")
    parser.add_argument("--data_path", type=str, default="/raid_sdh/home/xyg/RedPajama")
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--probe_depth", type=int, default=10)
    args = parser.parse_args()

    evaluate_model(args)