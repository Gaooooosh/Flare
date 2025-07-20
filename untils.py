
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import torchvision.transforms as T
from typing import List, Union, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

import datetime

class Logger:
    """
    一个带有时间戳和名称的logger类，用于将消息写入指定的日志文件
    """
    
    def __init__(self, log_file_path, logger_name="DefaultLogger"):
        """
        初始化Logger对象
        
        Args:
            log_file_path (str): 日志文件的路径
            logger_name (str): logger的名称，默认为"DefaultLogger"
        """
        self.log_file_path = log_file_path
        self.logger_name = logger_name
        
    def log(self, message, include_timestamp=True):
        """
        将消息写入日志文件，带有时间戳和logger名称
        
        Args:
            message (str): 要写入日志的消息
            include_timestamp (bool): 是否包含时间戳，默认为True
        """
        try:
            timestamp = ""
            if include_timestamp:
                timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            
            log_entry = f"{timestamp} [{self.logger_name}] {message}\n"
            
            with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(log_entry)
            return True
        except Exception as e:
            print(f"写入日志失败: {e}")
            return False
    
    def clear_log(self):
        """
        清空日志文件内容
        """
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write("")
            return True
        except Exception as e:
            print(f"清空日志失败: {e}")
            return False
    
    def set_logger_name(self, new_name):
        """
        修改logger的名称
        
        Args:
            new_name (str): 新的logger名称
        """
        self.logger_name = new_name

def load_model(
    model_path: str,
    device: Union[str, List[str], Dict[str, Union[int, str]]] = "cuda",
    use_flash_attn: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
    output_attention: Optional[bool] = False,
    **kwargs
) -> tuple:
    """
    加载语言模型和对应的tokenizer，支持分布式加载到多个GPU上
    
    Args:
        model_path: 模型路径
        device: 设备指定，可以是字符串("cuda:0")、设备列表(["cuda:0", "cuda:1"])或设备映射字典
        use_flash_attn: 是否使用Flash Attention 2
        torch_dtype: 模型权重的数据类型
        trust_remote_code: 是否信任远程代码
        skip_layers: 需要跳过的注意力层索引列表
        attn_implementation: 注意力实现方式，默认为None，当use_flash_attn=True时设为"flash_attention_2"
        patch_attn_func: 用于修改注意力机制的函数
        **kwargs: 传递给AutoModelForCausalLM.from_pretrained的其他参数
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载模型: {model_path}")
    
    # 加载tokenizer
    
    # 设置设备映射
    if isinstance(device, str):
        device_map = device
    elif isinstance(device, list):
        # 如果提供了GPU列表，创建均匀分布的设备映射
        if len(device) == 1:
            device_map = device[0]
        else:
            device_map = "auto"
            CUDA_VISIBLE_DEVICES = ",".join([d.split(":")[-1] if ":" in d else d for d in device])
            print(f"CUDA_VISIBLE_DEVICES:{CUDA_VISIBLE_DEVICES}")
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    else:
        # 如果是自定义的设备映射字典
        device_map = device
    
    # 设置模型加载配置
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs
    }

    
    # Flash Attention 设置
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("启用 Flash Attention 2")
    elif attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    return model, tokenizer


def grid_show(to_shows, cols,file_name):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*10.5, cols*4))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    # plt.show()
    plt.savefig(file_name)

def visualize_head(att_map,file_name):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # plt.show()
    plt.savefig(file_name)
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def visualize_layer(att_map, file_name):
    percentile=99
    cmap='viridis'
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    att_map = att_map.squeeze()
    att_map = att_map.sum(axis=0)
    # vmin, vmax = np.percentile(att_map, (1, percentile))
    im = ax.imshow(att_map, cmap = cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Attention Weight', rotation=270, labelpad=15)
    # 设置坐标轴标签
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Key Position')
    ax.set_title(f'Average Attention Map (Top {percentile}% Values)')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
