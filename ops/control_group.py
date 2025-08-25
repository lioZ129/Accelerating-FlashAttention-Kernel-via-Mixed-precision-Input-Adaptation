"""
对照组前向（先反量化再 FlashAttention）

要点：
- 提供 `flash_attention_forward_with_pre_dequantization(...)`：
  - 根据 `bits` 选择 INT8 或 INT4 的反量化封装；
  - 得到 `K_deq, V_deq` 后，调用 `flash_attention_fp16_forward`；
  - 保留参数：`kv_is_fp16_mask, K_fp16, V_fp16, causal, softmax_scale`
- 负责准备 strides、grid 并进行基本的输入校验（形状、位宽、HEAD_DIM 偶数约束等）

"""

from typing import Optional

import torch

from quant_flash_attn.kernels.flash_attn_fp16_fwd import (
    flash_attention_fp16_forward,
)
from quant_flash_attn.kernels.quant_dequant_int8 import (
    dequantize_kv_cache_from_int8,
)
from quant_flash_attn.kernels.quant_dequant_int4 import (
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
    assert bits in (4, 8)
    if bits == 8:
        K_deq, V_deq = dequantize_kv_cache_from_int8(
            K_q_or_packed, V_q_or_packed, scale_k, scale_v, kv_is_fp16_mask, K_fp16, V_fp16
        )
    else:
        K_deq, V_deq = dequantize_kv_cache_from_packed_int4(
            K_q_or_packed, V_q_or_packed, scale_k, scale_v, kv_is_fp16_mask, K_fp16, V_fp16
        )
    return flash_attention_fp16_forward(Q, K_deq, V_deq, causal=causal)



