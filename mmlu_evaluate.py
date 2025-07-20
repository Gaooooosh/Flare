import asyncio
import os
import argparse
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from untils import Logger
from patch_qwen_rope import patch_qwen_rope
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def parse_args():
    parser = argparse.ArgumentParser(description="测试模型在MMLU数据集上的表现")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument('--devices', nargs='+', type=int, help='GPU device IDs to use')
    parser.add_argument("--use_flash_attn", action="store_true", help="是否使用Flash Attention 2")
    parser.add_argument("--greedy", action="store_true", help="是否使用贪婪解码")
    parser.add_argument("--max_samples", type=int, default=-1, help="每个任务使用的最大样本数，-1表示使用全部")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--mmlu_path", type=str, default="/raid_sdh/home/xyg/mmlu", help="MMLU数据集的本地路径")
    return parser.parse_args()

def get_completed_tasks(result_dir):
    completed_tasks = set()
    if os.path.exists(result_dir):
        for filename in os.listdir(result_dir):
            if filename.endswith(".json"):
                task_name = filename.split(".")[0]
                completed_tasks.add(task_name)
    return completed_tasks

def save_results(result_dir, task_name, results):
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"{task_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

def save_summary(result_dir, summary):
    csv_path = os.path.join(result_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Samples"])
        for task, (acc, samples) in summary.items():
            writer.writerow([task, acc, samples])
    
    total_correct = sum([acc * samples for acc, samples in summary.values()])
    total_samples = sum([samples for _, samples in summary.values()])
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Average", avg_acc, total_samples])

def process_instruct_prompts(prompt, model_path):
    if "instruct" in model_path.lower():
        if "llama" in model_path.lower():
            return f"<|start_header_id|><|system|><|end_header_id|>According to the questions, answer question with options (A,B,C or D), then give the answers with only the Options, in one word.<|eot_id|><|start_header_id|><|user|><|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>The Answers is:"
        elif "qwen" in model_path.lower():
            return f"<|im_start|>system\n<|im_end|>According to the questions, answer question with options (A,B,C or D), then give the answers with only the Options, in one word.\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nThe Answers is:"
    return f"{prompt}\nThe Answers is:"

def evaluate_multiple_choice(model, tokenizer, dataset, args, logger):
    results = []
    correct = 0
    total = 0
    
    if args.max_samples > 0 and args.max_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.max_samples, replace=False)
        dataset = dataset.select(indices)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    generation_config = {
        "max_new_tokens": 1,
        "use_cache" : False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        generation_config["do_sample"] = False
        generation_config["temperature"] = 1.0
        generation_config["top_p"] = 1.0
        generation_config["num_beams"] = 1
    else:
        generation_config["do_sample"] = True
        generation_config["temperature"] = 0.8
        generation_config["top_p"] = 0.8

    for idx,batch in tqdm(enumerate(dataloader), desc="Evaluating"):
        questions = batch["question"]
        choices = batch["choices"]
        answers = batch["answer"]
        options = ["A", "B", "C", "D"]
        batch_results = []
        batch_correct = 0
        prompts = []
        for i, question in enumerate(questions):
            prompt = f"{question}\nA. {choices[0][i]}\nB. {choices[1][i]}\nC. {choices[2][i]}\nD. {choices[3][i]}\n"
            prompts.append(process_instruct_prompts(prompt, args.model_path))
        inputs = tokenizer(prompts, return_tensors="pt",padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs,** generation_config,num_return_sequences=1)
        
        generated_sequences = []
        for i, output in enumerate(outputs):
            generated_ids = output[-1:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()
            # 只取第一个字母作为预测结果
            predicted_answer = generated_text[0] if generated_text and generated_text[0] in ['A', 'B', 'C', 'D'] else ''
            generated_sequences.append(predicted_answer)
        
        batch_correct = 0
        for i,pred in enumerate(generated_sequences):
            score = 0.0
            is_correct = False
            correct_answer = options[answers[i]]
            predicted_answer = pred
            if predicted_answer == correct_answer:
                score = 1.0
                is_correct = True
                batch_correct += 1
            result = {
                "question": questions[i],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "score": score
            }
            batch_results.append(result)
        
        results.extend(batch_results)
        total += len(batch_results)
        accuracy = batch_correct / len(batch_results) if len(batch_results) > 0 else 0
        logger.log(f"Batch[{idx}]:finished,ACC:{accuracy}")
    return results, accuracy, total

def main():
    args = parse_args()
    
    result_dir = os.path.join("MMLU_result", f"{os.path.basename(args.model_path)}")
    os.makedirs(result_dir, exist_ok=True)
    logger = Logger(os.path.join(result_dir,"log.log"),"MMLU_LOG")
    completed_tasks = get_completed_tasks(result_dir)
    
    logger.log(f"加载模型: {args.model_path}")
    model_kwargs = {}
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    device_map = "auto"
    if args.devices:
        if len(args.devices) == 1:
            device_map = f"cuda:{args.devices[0]}"
        else:
            device_map = {i: f"cuda:{device}" for i, device in enumerate(args.devices)}
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
          output_attentions=True,
        device_map='cuda:0'
    )
    
    config_path = os.path.join(args.model_path, 'config.json')
    logger.log(f'Reading config from {config_path}')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # 应用猴子补丁
            print(f'Applying monkey patch with no_rope_layers: {config.get("nope_layers", [])}')
            patch_qwen_rope(model, no_rope_layers=config.get('nope_layers', []))
    except Exception as e:
        print(f'Error applying monkey patch: {e}')
    except ImportError as e:
        logger.log(f'Error importing flare.attn_patch: {e}')
    except Exception as e:
        logger.log(f'Error applying monkey patch: {e}')
    
    mmlu_path = os.path.abspath(args.mmlu_path)
    logger.log(f"加载MMLU数据集: {mmlu_path}")
    tasks = []
    if os.path.exists(mmlu_path):
        for filename in os.listdir(mmlu_path):
            task_path = os.path.join(mmlu_path, filename)
            if os.path.isdir(task_path):
                tasks.append((task_path, filename))
    
    logger.log(f"找到 {len(tasks)} 个任务")
    summary = {}
    
    for path,task in tasks:
        if task in completed_tasks:
            logger.log(f"跳过已完成的任务: {task}")
            with open(os.path.join(result_dir, f"{task}.json"), "r") as f:
                task_data = json.load(f)
                if "accuracy" in task_data and "total_samples" in task_data:
                    summary[task] = (task_data["accuracy"], task_data["total_samples"])
            continue
        
        logger.log(f"评估任务: {task}")
        try:
            test_dataset = load_from_disk(path)['test']
        except Exception as e:
            logger.log(f"加载路径：{path}错误: {e}")
            continue
        results, accuracy, total_samples = evaluate_multiple_choice(model, tokenizer, test_dataset, args, logger)
        
        task_results = {
            "task": task,
            "accuracy": accuracy,
            "total_samples": total_samples,
            "results": results
        }
        save_results(result_dir, task, task_results)
        summary[task] = (accuracy, total_samples)
        save_summary(result_dir, summary)
        logger.log(f"任务 {task} 完成，准确率: {accuracy:.4f}, 样本数: {total_samples}")
    
    save_summary(result_dir, summary)
    logger.log(f"所有任务评估完成，结果保存在 {result_dir}")

if __name__ == "__main__":
    main()