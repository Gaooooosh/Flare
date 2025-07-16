"""
patch_qwen_rope.py
功能：给 Qwen2 指定层禁用 RoPE（跳过旋转）
用法：
    from patch_qwen_rope import patch_qwen_rope
    patch_qwen_rope(model, no_rope_layers=[0, 1, 2])
"""

import torch
from typing import List, Optional, Tuple
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


def _patched_attn_forward(
    self: Qwen2Attention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    """改写后的 attention forward：指定层跳过 RoPE"""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 如果该层在 no_rope 列表里，则直接跳过旋转
    if not getattr(self, "_rope_disabled", False):
        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
    else:
        # 禁用 RoPE 时，sin/cos 仍传入 past_key_value 以兼容 KV-Cache
        if past_key_value is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # 选择 attention 实现（flash / sdpa / eager）
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )
    attention_interface = (
        ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation != "eager"
        else eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_qwen_rope(model, no_rope_layers: List[int]):
    """
    入口函数：给 model 的指定层禁用 RoPE
    :param model: transformers 加载的 Qwen2ForCausalLM
    :param no_rope_layers: 需要禁用 RoPE 的层号列表（0-based）
    """
    for idx, layer in enumerate(model.model.layers):
        if idx in no_rope_layers:
            # 标记并替换 forward
            layer.self_attn._rope_disabled = True
            layer.self_attn.forward = _patched_attn_forward.__get__(
                layer.self_attn, type(layer.self_attn)
            )