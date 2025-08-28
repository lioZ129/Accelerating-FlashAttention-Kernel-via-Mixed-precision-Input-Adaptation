"""
对照组：先反量化 + KV 常驻调度 FlashAttention

要点：
- 先反量化：根据 bits 选择 INT8/INT4 反量化，得到 FP16 的 K/V
- 后计算：使用与实验组相同的 KV 常驻调度 FlashAttention
- 对比：相同调度策略下，对比"先反量化"vs"即时反量化"的差异
"""

from typing import Optional

import torch

from kernels.flash_attn_fp16_fwd import (
    flash_attention_fp16_kv_persistent_forward,
)
from kernels.quant_dequant_int8 import (
    dequantize_kv_cache_from_int8,
)
from kernels.quant_dequant_int4 import (
    dequantize_kv_cache_from_packed_int4,
)


def flash_attention_forward_with_pre_dequantization(
    Q: torch.Tensor,
    K_q_or_packed: torch.Tensor,
    V_q_or_packed: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    *,
    bits: int = 8,
    kv_is_fp16_mask: Optional[torch.Tensor] = None,
    K_fp16: Optional[torch.Tensor] = None,
    V_fp16: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    assert Q.is_cuda and K_q_or_packed.is_cuda and V_q_or_packed.is_cuda
    assert bits in (4, 8), f"不支持的位宽: {bits}"
    
    if bits == 4:
        assert Q.shape[-1] % 2 == 0, "INT4 量化要求 HEAD_DIM 为偶数"
    
    # 先反量化
    if bits == 8:
        K_deq, V_deq = dequantize_kv_cache_from_int8(
            K_q_or_packed, V_q_or_packed, scale_k, scale_v, 
            kv_is_fp16_mask, K_fp16, V_fp16
        )
    else:  # bits == 4
        K_deq, V_deq = dequantize_kv_cache_from_packed_int4(
            K_q_or_packed, V_q_or_packed, scale_k, scale_v, 
            kv_is_fp16_mask, K_fp16, V_fp16
        )
    
    # 再使用相同的 KV 常驻调度
    return flash_attention_fp16_kv_persistent_forward(Q, K_deq, V_deq, causal=causal)