"""
使用niah探针实验评估模型
支持：
1. 配置测试的上下文长度和探针的深度
2. 汇报所有层的注意力模式
3. 分层、分头保存注意力分数

作者：xyg
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import json
import pandas as pd
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from niah_probe import NIAHProbe
from patch_qwen_rope import patch_qwen_rope
from datasets import load_dataset
from tqdm import tqdm
import random
# 定义评估参数的数据类
@dataclass
class EvaluationArguments:
    model_name_or_path: str = field(default="/raid_sdh/home/xyg/output_qwen3b_redpajama_nope-20~")
    data_path: str = field(default="/raid_sdh/home/xyg/RedPajama")
    max_seq_len: int = field(default=32768)
    probe_position: int = field(default=10)  # 合并probe_depth功能
    fixed_text_length: int = field(default=32768)  # 固定测试文本长度

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
    import re
    
    # 初始化正确率计算变量
    correct_predictions = 0
    total_predictions = 0
    print(f'Applying monkey patch with no_rope_layers: {config.get("nope_layers", [])}')
    try:
        patch_qwen_rope(model, no_rope_layers=config.get('nope_layers', []))
    except Exception as e:
        print(f'Error applying monkey patch: {e}')
        return

    # 加载数据集
    data_files = eval_args.data_path
    print(f"Loading data from: {data_files}")
    raw_dataset = load_dataset("json", data_files=data_files, split="train")
    eval_dataset = raw_dataset.select(range(10))

    # 插入magic number到每个样本的指定深度
    MAGIC_NUMBER = random.randint(10000, 99999)  # 定义magic number作为探针
    # 截断原始文本到固定长度，然后插入magic number
    eval_dataset = eval_dataset.map(lambda x: {"text": f"请找出这段文本中的magic number：{x.get('raw_content', x.get('content', x.get('document', '')))[:eval_args.probe_position]} magic number:{MAGIC_NUMBER} {x.get('raw_content', x.get('content', x.get('document', '')))[eval_args.probe_position:eval_args.fixed_text_length]} The magic number is:"})

    # 初始化探针（将magic number转换为token ID用于检测）
    magic_token_id = tokenizer(str(MAGIC_NUMBER), add_special_tokens=False)['input_ids'][0]
    probe = NIAHProbe(model, tokenizer, max_seq_len=eval_args.max_seq_len, probe_position=eval_args.probe_position, magic_token_id=magic_token_id)

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
    magic_scores = {}  # 新增：记录对magic number的注意力分数
    
    for layer in range(model.config.num_hidden_layers):
        sink_score = []
        probe_score = []
        other_score = []
        magic_score = []
        
        for batch in tqdm(eval_dataset):
            input_ids = batch['input_ids'].unsqueeze(0).to(model.device)
            mask = batch['attention_mask'].unsqueeze(0).to(model.device)
            attention_scores = probe.get_attention_scores(input_ids)
            # 生成模型回答
            outputs = model.generate(
                input_ids,
                attention_mask=mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            # print(generated_text)
            # 记录输入样本和模型响应到JSON日志文件
            original_text = tokenizer.decode(batch['input_ids'], skip_special_tokens=True)
            log_entry = {
                "input": original_text,
                "response": generated_text
            }
            with open("model_responses.jsonl", "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
            # 提取预测的magic number
            predicted_num = None
            match = re.search(r'\b\d+\b', generated_text)
            if match:
                predicted_num = int(match.group())

            # 验证答案（假设样本中正确答案存储在sample['magic_number']）
            expected_num = MAGIC_NUMBER
            if expected_num is not None and predicted_num == expected_num:
                correct_predictions += 1
            total_predictions += 1
            layer_data = attention_scores.get(layer, {})
            # 从探针结果中获取当前层的分数
            
            # 计算sink token注意力得分
            if 'sink' in layer_data:
                sink_score.append(sum(layer_data['sink']) / len(layer_data['sink']))
            
            # 计算探针注意力得分
            if eval_args.probe_position > 0 and 'probe' in layer_data:
                probe_score.append(sum(layer_data['probe']) / len(layer_data['probe']))
            
            # 计算其他token注意力得分
            if 'other' in layer_data:
                other_score.append(sum(layer_data['other']) / len(layer_data['other']) if layer_data['other'] else 0)
            
            # 新增：计算对magic number的注意力得分
            if 'magic' in layer_data:
                magic_score.append(sum(layer_data['magic']) / len(layer_data['magic']) if layer_data['magic'] else 0)
        
        sink_scores[layer] = sum(sink_score) / len(sink_score) if sink_score else 0
        if eval_args.probe_position > 0:
            probe_scores[layer] = sum(probe_score) / len(probe_score) if probe_score else 0
        other_scores[layer] = sum(other_score) / len(other_score) if other_score else 0
        magic_scores[layer] = sum(magic_score) / len(magic_score) if magic_score else 0

    # 汇报结果
    for layer in range(model.config.num_hidden_layers):
        avg_sink = sink_scores.get(layer, 0)
        avg_magic = magic_scores.get(layer, 0)
        
        if eval_args.probe_position > 0:
            avg_probe = probe_scores.get(layer, 0)
            avg_other = other_scores.get(layer, 0)
            print(f"Layer {layer}: sink={avg_sink:.4f}, probe={avg_probe:.4f}, magic={avg_magic:.4f}, other={avg_other:.4f}")
        else:
            print(f"Layer {layer}: sink={avg_sink:.4f}, magic={avg_magic:.4f}")

    # 计算并输出正确率
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nModel Accuracy on Magic Number Prediction: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    else:
        print("\nNo predictions were made during evaluation.")

    # 保存注意力分数
    results = []
    for layer in sink_scores:
        result = {
            'Layer': layer,
            'Sink_Score': sink_scores[layer],
            'Magic_Score': magic_scores[layer],
            'Other_Score': other_scores[layer]
        }
        if eval_args.probe_position > 0:
            result['Probe_Score'] = probe_scores[layer]
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv("attention_scores.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/raid_sdh/home/xyg/output_qwen3b_redpajama_allrope")
    parser.add_argument("--data_path", type=str, default="/raid_sdh/home/xyg/RedPajama/sample/documents/2023-06/0003/en_middle.json.gz")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--probe_position", type=int, default=2048)  # 合并probe_position功能
    parser.add_argument("--fixed_text_length", type=int, default=4096, help="固定测试文本的长度")
    args = parser.parse_args()

    evaluate_model(args)