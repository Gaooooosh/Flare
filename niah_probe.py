"""
niah_probe.py
功能：实现NIAH探针实验，用于评估模型注意力模式
"""
import torch
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class NIAHProbe:
    """
    NIAH探针实验类
    用于计算模型各层各头的注意力分数
    """
    
    def __init__(self, model, tokenizer, max_seq_len: int = 32768, probe_depth: int = 10):
        """
        初始化探针
        :param model: 待评估的模型
        :param tokenizer: 分词器
        :param max_seq_len: 最大序列长度
        :param probe_depth: 探针深度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.probe_depth = probe_depth
    
    def get_attention_scores(self, input_ids: torch.Tensor) -> Dict[int, Dict[int, List[float]]]:
        """
        获取注意力分数
        :param input_ids: 输入token IDs
        :return: 各层各头的注意力分数字典
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True, return_dict=True)

        if outputs.attentions is None:
            raise ValueError("Model did not return attention weights. Check if output_attentions is enabled in model configuration.")

        attention_scores = {}
        for layer_idx, layer_attentions in enumerate(outputs.attentions):
            if layer_attentions is None:
                raise ValueError(f"Attention weights for layer {layer_idx} are None. Check model configuration.")
            # 取最后一个token的注意力分数 (batch=0, head=all)
            layer_scores = layer_attentions[0, :, -1, :].mean(dim=0).tolist()
            
            # 分类计算三种注意力分数
            attention_scores[layer_idx] = {
                "sink": layer_scores[:10],  # 前10个token (sink tokens)
                "probe": layer_scores[10:10+self.probe_depth],  # 探针token
                "other": layer_scores[10+self.probe_depth:]  # 其他无关token
            }
            
        return attention_scores
    
    def calculate_average_scores(self, attention_scores: Dict[int, Dict[int, List[float]]]) -> Dict[str, Dict[int, float]]:
        """
        计算平均注意力分数
        :param attention_scores: 原始注意力分数
        :return: 各层各类token的平均分数
        """
        avg_scores = {
            "sink": {},
            "probe": {},
            "other": {}
        }
        
        for layer_idx, scores in attention_scores.items():
            avg_scores["sink"][layer_idx] = sum(scores["sink"]) / len(scores["sink"]) if scores["sink"] else 0
            avg_scores["probe"][layer_idx] = sum(scores["probe"]) / len(scores["probe"]) if scores["probe"] else 0
            avg_scores["other"][layer_idx] = sum(scores["other"]) / len(scores["other"]) if scores["other"] else 0
            
        return avg_scores