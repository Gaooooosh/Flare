"""
niah_probe.py
功能：实现NIAH探针实验，用于评估模型注意力模式
"""
import torch
from typing import Dict, List
from dataclasses import dataclass
from typing import Optional
@dataclass
class NIAHProbe:
    """
    NIAH探针实验类
    用于计算模型各层各头的注意力分数
    """
    
    def __init__(self, model, tokenizer, max_seq_len: int = 32768, probe_position: int = 10, magic_token_id: Optional[int] = None):
        """
        初始化探针
        :param model: 待评估的模型
        :param tokenizer: 分词器
        :param max_seq_len: 最大序列长度
        :param probe_depth: 探针深度
        :param magic_token_id: 用于检测的magic number对应的token ID
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.probe_position = probe_position
        self.magic_token_id = magic_token_id
    
    def get_attention_scores(self, input_ids: torch.Tensor) -> Dict[int, Dict[str, List[float]]]:
        """
        获取注意力分数
        :param input_ids: 输入token IDs
        :return: 各层各头的注意力分数字典，包含magic token的注意力分数
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True, return_dict=True)

        if outputs.attentions is None:
            raise ValueError("Model did not return attention weights. Check if output_attentions is enabled in model configuration.")

        attention_scores = {}
        # 查找magic token在输入序列中的位置（取第一个出现的位置）
        magic_positions = (input_ids == self.magic_token_id).nonzero(as_tuple=True)[1].tolist() if self.magic_token_id is not None else []
        magic_position = magic_positions[0] if magic_positions else -1

        for layer_idx, layer_attentions in enumerate(outputs.attentions):
            if layer_attentions is None:
                raise ValueError(f"Attention weights for layer {layer_idx} are None. Check model configuration.")
            # 取最后一个token的注意力分数 (batch=0, head=all)
            layer_scores = layer_attentions[0, :, -1, :].mean(dim=0).tolist()
            
            # 分类计算注意力分数
            if 0 <= magic_position < len(layer_scores):
                  probe_start = max(0, magic_position - 5)
                  probe_end = min(len(layer_scores), magic_position + 5)
                  layer_data = {
                      "sink": layer_scores[:10],  # 前10个token (sink tokens)
                      "probe": layer_scores[probe_start:probe_end],  # 探针区域（围绕magic token）
                      "other": [score for i, score in enumerate(layer_scores) if (i < probe_start or i >= probe_end) and i >= 10],  # 排除sink和probe的其他token
                      "magic": [layer_scores[magic_position]]  # magic token自身分数
                  }
            else:
                  # 没有找到magic token时使用动态划分
                  if len(layer_scores) > 20 + self.probe_position:
                      # 使用中间区域作为探针（避开前10个sink token）
                      probe_start = (len(layer_scores) - self.probe_position) // 2
                      probe_end = probe_start + self.probe_position
                      layer_data = {
                          "sink": layer_scores[:10],  # 前10个token (sink tokens)
                          "probe": layer_scores[probe_start:probe_end],  # 中间区域作为探针
                          "other": [score for i, score in enumerate(layer_scores) if (i < probe_start or i >= probe_end) and i >= 10],  # 排除sink和probe的其他token
                          "magic": []  # 无magic token
                      }
                  else:
                      # 输入过短时使用原始划分
                      layer_data = {
                          "sink": layer_scores[:10],
                          "probe": layer_scores[10:10+self.probe_position],
                          "other": layer_scores[10+self.probe_position:],
                          "magic": []
                      }
            
            attention_scores[layer_idx] = layer_data
            
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