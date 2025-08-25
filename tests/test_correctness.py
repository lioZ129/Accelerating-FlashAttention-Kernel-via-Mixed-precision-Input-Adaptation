"""
正确性测试（参考实现对比 + 两组互相对比）

说明：
- 参考实现采用 PyTorch：先得到“反量化后的 FP16” K/V，再计算 softmax(QK^T)*V。
- 对照组：先反量化再调用 FP16 FA；实验组：on-the-fly 内核内即时反量化。
"""

from typing import Tuple

import argparse
import torch

from quant_flash_attn.utils.quantize import (
    quantize_kv_cache_fp16_to_int8,
    quantize_kv_cache_fp16_to_packed_int4,
    make_mixed_precision_kv_mask,
)
from quant_flash_attn.kernels.quant_dequant_int8 import (
    dequantize_kv_cache_from_int8,
)
from quant_flash_attn.kernels.quant_dequant_int4 import (
    dequantize_kv_cache_from_packed_int4,
)
from quant_flash_attn.ops.control_group import (
    flash_attention_forward_with_pre_dequantization,
)
from quant_flash_attn.kernels.flash_attn_quant_onthefly import (
    flash_attention_forward_with_on_the_fly_dequantization,
)


def compute_reference_output_with_fp16_kv(
    Q: torch.Tensor, K_fp16: torch.Tensor, V_fp16: torch.Tensor, causal: bool
) -> torch.Tensor:
    """PyTorch 参考实现，基于 FP16 K/V"""
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)
    P = torch.matmul(Q, K_fp16.transpose(2, 3)) * softmax_scale
    if causal:
        mask = torch.tril(torch.ones((S, S), device=Q.device, dtype=torch.bool))
        P[:, :, ~mask] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).to(Q.dtype)
    return torch.matmul(P, V_fp16)


def _gen_data(
    B: int, H: int, S: int, D: int, bits: int, causal: bool, fp16_ratio: float, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成随机 Q/K/V 与量化数据、掩码"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16, generator=g)
    K_fp16 = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16, generator=g)
    V_fp16 = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16, generator=g)
    mask = make_mixed_precision_kv_mask((B, H, S), fp16_ratio).to(device=Q.device)

    if bits == 8:
        K_q, V_q, scale_k, scale_v = quantize_kv_cache_fp16_to_int8(K_fp16, V_fp16)
        return Q, K_fp16, V_fp16, K_q, V_q, scale_k, scale_v, mask
    else:
        K_qp, V_qp, scale_k, scale_v = quantize_kv_cache_fp16_to_packed_int4(K_fp16, V_fp16)
        return Q, K_fp16, V_fp16, K_qp, V_qp, scale_k, scale_v, mask


def verify_correctness_int8_against_reference(
    B: int, H: int, S: int, D: int, causal: bool, fp16_ratio: float
):
    Q, K_fp16, V_fp16, K_q, V_q, scale_k, scale_v, mask = _gen_data(B, H, S, D, 8, causal, fp16_ratio)
    # 参考 K/V：使用对照组的反量化结果以保证一致的量化误差基准
    K_deq_ref, V_deq_ref = dequantize_kv_cache_from_int8(K_q, V_q, scale_k, scale_v, mask, K_fp16, V_fp16)
    O_ref = compute_reference_output_with_fp16_kv(Q, K_deq_ref, V_deq_ref, causal)

    # 对照组
    O_ctrl = flash_attention_forward_with_pre_dequantization(
        Q, K_q, V_q, scale_k, scale_v, bits=8, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=causal
    )
    # 实验组
    O_exp = flash_attention_forward_with_on_the_fly_dequantization(
        Q, K_q, V_q, scale_k, scale_v, bits=8, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=causal
    )

    atol, rtol = 1e-2, 0.0
    assert torch.allclose(O_ref, O_ctrl, atol=atol, rtol=rtol)
    assert torch.allclose(O_ref, O_exp, atol=atol, rtol=rtol)


def verify_correctness_int4_against_reference(
    B: int, H: int, S: int, D: int, causal: bool, fp16_ratio: float
):
    assert D % 2 == 0, "INT4 测试要求 D 为偶数"
    Q, K_fp16, V_fp16, K_qp, V_qp, scale_k, scale_v, mask = _gen_data(B, H, S, D, 4, causal, fp16_ratio)
    K_deq_ref, V_deq_ref = dequantize_kv_cache_from_packed_int4(K_qp, V_qp, scale_k, scale_v, mask, K_fp16, V_fp16)
    O_ref = compute_reference_output_with_fp16_kv(Q, K_deq_ref, V_deq_ref, causal)

    O_ctrl = flash_attention_forward_with_pre_dequantization(
        Q, K_qp, V_qp, scale_k, scale_v, bits=4, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=causal
    )
    O_exp = flash_attention_forward_with_on_the_fly_dequantization(
        Q, K_qp, V_qp, scale_k, scale_v, bits=4, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=causal
    )

    atol, rtol = 1e-2, 0.0
    assert torch.allclose(O_ref, O_ctrl, atol=atol, rtol=rtol)
    assert torch.allclose(O_ref, O_exp, atol=atol, rtol=rtol)


def main_run_correctness_checks():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--fp16-ratio", type=float, default=0.0)
    args = parser.parse_args()

    torch.cuda.synchronize()
    if args.bits == 8:
        verify_correctness_int8_against_reference(
            args.batch_size, args.num_heads, args.seq_len, args.head_dim, args.causal, args.fp16_ratio
        )
    else:
        verify_correctness_int4_against_reference(
            args.batch_size, args.num_heads, args.seq_len, args.head_dim, args.causal, args.fp16_ratio
        )
    print("[correctness] PASS")


if __name__ == "__main__":
    main_run_correctness_checks()



