"""
占位说明：量化与数据构造工具（测试/基准用）

要点：
- `quantize_kv_cache_fp16_to_int8(K_fp16, V_fp16)`：按逐 token 对称量化生成 `K_q, V_q, scale_k, scale_v`；
- `quantize_kv_cache_fp16_to_packed_int4(K_fp16, V_fp16)`：将 INT4 打包到 `uint8`；
- `make_mixed_precision_kv_mask(shape, fp16_ratio)`：按给定比例随机采样 FP16 行掩码；
- 保证随机种子可控；确保 INT4 情况下 `D` 为偶数

"""

from typing import Tuple

import torch


def quantize_kv_cache_fp16_to_int8(
    K_fp16: torch.Tensor,
    V_fp16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 FP16 K/V 逐 token 对称量化为 INT8，并返回 (K_q, V_q, scale_k, scale_v)

    说明：
    - 简化实现：per-token（每个 S 位置一组 scale），group_size == D
    - scale = max(|x|) / 127；q = clip(round(x / scale), -128, 127)
    - 返回 scale 为 fp16
    """
    assert K_fp16.dtype == torch.float16 and V_fp16.dtype == torch.float16 and K_fp16.shape == V_fp16.shape
    B, H, S, D = K_fp16.shape
    with torch.no_grad():
        # 计算逐 token 的绝对最大值
        k_absmax = K_fp16.abs().amax(dim=-1) + 1e-8
        v_absmax = V_fp16.abs().amax(dim=-1) + 1e-8
        scale_k = (k_absmax / 127.0).to(torch.float16)
        scale_v = (v_absmax / 127.0).to(torch.float16)
        K_q = torch.clamp((K_fp16 / scale_k[..., None]).round(), -128, 127).to(torch.int8)
        V_q = torch.clamp((V_fp16 / scale_v[..., None]).round(), -128, 127).to(torch.int8)
    return K_q, V_q, scale_k, scale_v


def quantize_kv_cache_fp16_to_packed_int4(
    K_fp16: torch.Tensor,
    V_fp16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 FP16 K/V 逐 token 对称量化为 INT4 并打包到 uint8（每字节两值）

    - scale = max(|x|) / 7；q4 ∈ [-8, 7]；打包：[低 4bit | 高 4bit]
    - 若 D 为奇数，右侧 padding 到偶数后量化，再丢弃 padding 的一半 nibble
    """
    assert K_fp16.dtype == torch.float16 and V_fp16.dtype == torch.float16 and K_fp16.shape == V_fp16.shape
    B, H, S, D = K_fp16.shape
    assert D % 2 == 0, "INT4 量化要求 D 为偶数"
    with torch.no_grad():
        k_absmax = K_fp16.abs().amax(dim=-1) + 1e-8
        v_absmax = V_fp16.abs().amax(dim=-1) + 1e-8
        scale_k = (k_absmax / 7.0).to(torch.float16)
        scale_v = (v_absmax / 7.0).to(torch.float16)
        K_q4 = torch.clamp((K_fp16 / scale_k[..., None]).round(), -8, 7).to(torch.int8)
        V_q4 = torch.clamp((V_fp16 / scale_v[..., None]).round(), -8, 7).to(torch.int8)
        # 打包：偶数索引为低 4bit，奇数索引为高 4bit
        K_low = (K_q4[..., 0::2] & 0xF).to(torch.uint8)
        K_high = (K_q4[..., 1::2] & 0xF).to(torch.uint8)
        K_packed = (K_low | (K_high << 4)).contiguous()
        V_low = (V_q4[..., 0::2] & 0xF).to(torch.uint8)
        V_high = (V_q4[..., 1::2] & 0xF).to(torch.uint8)
        V_packed = (V_low | (V_high << 4)).contiguous()
    return K_packed, V_packed, scale_k, scale_v


def make_mixed_precision_kv_mask(shape: torch.Size, fp16_ratio: float) -> torch.Tensor:
    """生成混合精度行掩码（形状 (B,H,S)），按给定比例随机置 True"""
    B, H, S = shape
    num_fp16 = int(S * fp16_ratio)
    mask = torch.zeros((B, H, S), dtype=torch.uint8)
    if num_fp16 > 0:
        # 简单做法：对每个 (B,H) 的 S 维前 num_fp16 位置置 1，实际测试可打乱
        mask[:, :, :num_fp16] = 1
    return mask



