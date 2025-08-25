"""
INT8 反量化内核与封装（对照组用）

要点：
- `@triton.jit` 实现 `_dequantize_kv_cache_rows_from_int8_kernel`：
  - 输入：`K_q/V_q`（int8）、`scale_k/scale_v`（fp16/fp32）、可选 `kv_is_fp16_mask` 与 `K_fp16/V_fp16`；
  - 行级处理：按行读取，量化行执行 `fp = int8 * scale`，掩码为 True 的行从 `K_fp16/V_fp16` 直拷；
  - 输出：`K_deq/V_deq`（fp16）
- Python 封装 `dequantize_kv_cache_from_int8(...)`：
  - 组织网格/stride，调用 kernel；
  - 返回 `K_deq, V_deq`
- 尽量保证合并访存与批量向量化；`scale` 使用 fp16 存、计算前升格

"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_kv_cache_rows_from_int8_kernel(
    SRC,           # int8 源 (B,H,S,D)
    DST,           # fp16 目标 (B,H,S,D)
    SCALE,         # fp16/fp32 每 token 缩放 (B,H,S)
    SRC_FP16_OPT,  # 可选 FP16 源 (B,H,S,D)
    MASK_OPT,      # 可选掩码 (B,H,S) 非 0 表示使用 FP16 直拷
    # strides: 源/目标/可选源/scale/mask 均按 B,H,S,(D) 布局
    stride_src_b, stride_src_h, stride_src_s, stride_src_d,
    stride_dst_b, stride_dst_h, stride_dst_s, stride_dst_d,
    stride_fp16_b, stride_fp16_h, stride_fp16_s, stride_fp16_d,
    stride_scale_b, stride_scale_h, stride_scale_s,
    stride_mask_b, stride_mask_h, stride_mask_s,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    row = pid_row
    row_valid = row < SEQ_LEN

    # 计算行基址
    src_row = SRC + b * stride_src_b + h * stride_src_h + row * stride_src_s
    dst_row = DST + b * stride_dst_b + h * stride_dst_h + row * stride_dst_s
    fp16_row = SRC_FP16_OPT + b * stride_fp16_b + h * stride_fp16_h + row * stride_fp16_s
    scale_ptr = SCALE + b * stride_scale_b + h * stride_scale_h + row * stride_scale_s
    mask_ptr = MASK_OPT + b * stride_mask_b + h * stride_mask_h + row * stride_mask_s

    # 读取该行是否 FP16 直拷
    use_fp16 = tl.where(row_valid, tl.load(mask_ptr, mask=row_valid, other=0) > 0, False)
    scale_val = tl.load(scale_ptr, mask=row_valid, other=0.0).to(tl.float32)

    for start_d in range(0, HEAD_DIM, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        d_mask = row_valid & (offs_d < HEAD_DIM)
        src_ptrs = src_row + offs_d * stride_src_d
        dst_ptrs = dst_row + offs_d * stride_dst_d
        fp16_ptrs = fp16_row + offs_d * stride_fp16_d

        if use_fp16:
            vals = tl.load(fp16_ptrs, mask=d_mask, other=0.0)
            tl.store(dst_ptrs, vals, mask=d_mask)
        else:
            q_i8 = tl.load(src_ptrs, mask=d_mask, other=0).to(tl.int8)
            q_f = q_i8.to(tl.float32) * scale_val
            tl.store(dst_ptrs, q_f.to(tl.float16), mask=d_mask)


def dequantize_kv_cache_from_int8(
    K_q: torch.Tensor,
    V_q: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    kv_is_fp16_mask: Optional[torch.Tensor] = None,
    K_fp16: Optional[torch.Tensor] = None,
    V_fp16: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert K_q.is_cuda and V_q.is_cuda and K_q.dtype == torch.int8 and V_q.dtype == torch.int8
    assert K_q.shape == V_q.shape
    B, H, S, D = K_q.shape
    assert scale_k.shape == (B, H, S) and scale_v.shape == (B, H, S)

    if kv_is_fp16_mask is None:
        # 若无混合精度，则提供全 0 掩码与虚指针
        kv_is_fp16_mask = torch.zeros((B, H, S), device=K_q.device, dtype=torch.uint8)
        K_fp16 = torch.empty(0, device=K_q.device, dtype=torch.float16)
        V_fp16 = torch.empty(0, device=K_q.device, dtype=torch.float16)
        stride_fp16 = (0, 0, 0, 0)
    else:
        assert K_fp16 is not None and V_fp16 is not None
        assert K_fp16.shape == (B, H, S, D) and V_fp16.shape == (B, H, S, D)
        stride_fp16 = (K_fp16.stride(0), K_fp16.stride(1), K_fp16.stride(2), K_fp16.stride(3))

    K_deq = torch.empty((B, H, S, D), device=K_q.device, dtype=torch.float16)
    V_deq = torch.empty((B, H, S, D), device=V_q.device, dtype=torch.float16)

    grid = (S, B * H)
    BLOCK_D = 128

    _dequantize_kv_cache_rows_from_int8_kernel[grid](
        K_q, K_deq, scale_k, K_fp16 if K_fp16 is not None else K_deq, kv_is_fp16_mask,
        K_q.stride(0), K_q.stride(1), K_q.stride(2), K_q.stride(3),
        K_deq.stride(0), K_deq.stride(1), K_deq.stride(2), K_deq.stride(3),
        stride_fp16[0], stride_fp16[1], stride_fp16[2], stride_fp16[3],
        scale_k.stride(0), scale_k.stride(1), scale_k.stride(2),
        kv_is_fp16_mask.stride(0), kv_is_fp16_mask.stride(1), kv_is_fp16_mask.stride(2),
        NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D, BLOCK_D=BLOCK_D,
    )

    _dequantize_kv_cache_rows_from_int8_kernel[grid](
        V_q, V_deq, scale_v, V_fp16 if V_fp16 is not None else V_deq, kv_is_fp16_mask,
        V_q.stride(0), V_q.stride(1), V_q.stride(2), V_q.stride(3),
        V_deq.stride(0), V_deq.stride(1), V_deq.stride(2), V_deq.stride(3),
        stride_fp16[0], stride_fp16[1], stride_fp16[2], stride_fp16[3],
        scale_v.stride(0), scale_v.stride(1), scale_v.stride(2),
        kv_is_fp16_mask.stride(0), kv_is_fp16_mask.stride(1), kv_is_fp16_mask.stride(2),
        NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D, BLOCK_D=BLOCK_D,
    )

    return K_deq, V_deq



