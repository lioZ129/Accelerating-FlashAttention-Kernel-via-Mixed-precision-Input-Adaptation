"""
on-the-fly 反量化算子实现

要点：
- `@triton.jit` 实现 `_flash_attention_quantized_forward_inner_with_on_the_fly_dequantization`：
  - 支持 `BITS: tl.constexpr in {4,8}` 两种位宽的编译期特化；
  - 对每个 tile 的 K/V 行，按 `kv_is_fp16_mask` 行级选择：
    - FP16 行：直接 `tl.load`；
    - 量化行：INT8 直接乘 `scale`；INT4 拆 nibble 并符号扩展后乘 `scale`；
  - 构造统一的 `K_block/V_block`，与 Q 计算 `QK`、`P×V`，维持 log-sum-exp 稳定化
- `@triton.jit` 实现 `_flash_attention_quantized_forward_with_on_the_fly_dequantization` 并 autotune（按 `SEQ_LEN, HEAD_DIM, BITS` 作为 key）
- 提供 `FlashAttentionQuantizedForwardFunction` 与 `flash_attention_forward_with_on_the_fly_dequantization` 封装，组织 strides/grid，并选择 BITS 版本

"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_KV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_KV": 64}, num_warps=4, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_quantized_forward_with_on_the_fly_dequantization(
    Q,  # (B,H,S,D) fp16
    K_q_or_packed,  # int8 或 uint8 packed
    V_q_or_packed,  # int8 或 uint8 packed
    SCALE_K,  # (B,H,S) fp16/fp32
    SCALE_V,  # (B,H,S) fp16/fp32
    K_FP16_OPT,  # 可选 (B,H,S,D) fp16 源（K）
    V_FP16_OPT,  # 可选 (B,H,S,D) fp16 源（V）
    MASK_OPT,  # (B,H,S) uint8/bool，1 表示该行使用 FP16 直读
    O,  # 输出 (B,H,S,D) fp16
    softmax_scale,
    # strides Q/KV_packed/scale/mask/O
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_d_or_packed,
    stride_V_b, stride_V_h, stride_V_s, stride_V_d_or_packed,
    stride_KFP16_b, stride_KFP16_h, stride_KFP16_s, stride_KFP16_d,
    stride_VFP16_b, stride_VFP16_h, stride_VFP16_s, stride_VFP16_d,
    stride_SC_b, stride_SC_h, stride_SC_s,
    stride_MS_b, stride_MS_h, stride_MS_s,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BITS: tl.constexpr,
):
    """
    说明：
    - 在前向内核内按行即时反量化 K/V：BITS=8 直接 int8×scale；BITS=4 拆 nibble 后符号扩展×scale
    - 支持行级混合精度掩码：为 True 的行直接从 FP16 源读取；否则走量化路径
    - 其它流程与 FP16 版本一致：log-sum-exp 稳定化，P×V 累加
    """
    pid_q_blk = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    offs_q = pid_q_blk * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    offs_d = tl.arange(0, HEAD_DIM)

    # 基址
    q_base = b * stride_Q_b + h * stride_Q_h
    k_base = b * stride_K_b + h * stride_K_h
    v_base = b * stride_V_b + h * stride_V_h
    o_base = b * stride_O_b + h * stride_O_h
    kfp16_base = b * stride_KFP16_b + h * stride_KFP16_h
    vfp16_base = b * stride_VFP16_b + h * stride_VFP16_h
    sk_base = b * stride_SC_b + h * stride_SC_h
    sv_base = b * stride_SC_b + h * stride_SC_h
    ms_base = b * stride_MS_b + h * stride_MS_h

    # 载入 Q 块
    q_ptrs = Q + q_base + offs_q[:, None] * stride_Q_s + offs_d[None, :] * stride_Q_d
    Q_block = tl.load(q_ptrs, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    m_i = tl.full([BLOCK_SIZE_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_SIZE_Q], 1.0, dtype=tl.float32)
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        curr_kv = start_kv + offs_kv

        # 读 scale 与 mask（按行广播）
        sk_ptrs = SCALE_K + sk_base + curr_kv * stride_SC_s
        sv_ptrs = SCALE_V + sv_base + curr_kv * stride_SC_s
        ms_ptrs = MASK_OPT + ms_base + curr_kv * stride_MS_s
        sk_val = tl.load(sk_ptrs, mask=(curr_kv < SEQ_LEN), other=0.0).to(tl.float32)
        sv_val = tl.load(sv_ptrs, mask=(curr_kv < SEQ_LEN), other=0.0).to(tl.float32)
        use_fp16_row = tl.load(ms_ptrs, mask=(curr_kv < SEQ_LEN), other=0).to(tl.int32) > 0

        # 构造 K^T 块 (D, BLOCK_KV)
        # 对每一列 d，我们需要对 BLOCK_KV 行做选择（FP16 or 量化）并取值
        # 先尝试量化路径读取
        if BITS == 8:
            # K: int8 -> fp32 * scale
            kT_ptrs = K_q_or_packed + k_base + offs_d[:, None] * stride_K_d_or_packed + curr_kv[None, :] * stride_K_s
            k_val_q = tl.load(kT_ptrs, mask=(offs_d[:, None] < HEAD_DIM) & (curr_kv[None, :] < SEQ_LEN), other=0).to(tl.int8)
            K_T_block = (k_val_q.to(tl.float32) * sk_val[None, :]).to(tl.float16)
        else:
            # BITS == 4, packed uint8，选择低/高 nibble
            pack_col = offs_d // 2
            is_low = (offs_d % 2) == 0
            k_pack_ptrs = K_q_or_packed + k_base + pack_col[:, None] * stride_K_d_or_packed + curr_kv[None, :] * stride_K_s
            packed_u8 = tl.load(k_pack_ptrs, mask=(pack_col[:, None] < (HEAD_DIM // 2)) & (curr_kv[None, :] < SEQ_LEN), other=0).to(tl.uint8)
            low_n = packed_u8 & 0xF
            high_n = (packed_u8 >> 4) & 0xF
            n = tl.where(is_low[:, None], low_n, high_n).to(tl.int16)
            n_signed = tl.where(n < 8, n, n - 16).to(tl.float32)
            K_T_block = (n_signed * sk_val[None, :]).to(tl.float16)

        # 覆盖 FP16 行（使用掩码避免不必要读取）
        kfp16_ptrs = K_FP16_OPT + kfp16_base + curr_kv[None, :] * stride_KFP16_s + offs_d[:, None] * stride_KFP16_d
        K_T_block_fp16 = tl.load(
            kfp16_ptrs,
            mask=(curr_kv[None, :] < SEQ_LEN) & (offs_d[:, None] < HEAD_DIM) & use_fp16_row[None, :],
            other=0.0,
        )
        K_T_block = tl.where(use_fp16_row[None, :], K_T_block_fp16, K_T_block)

        # QK 并缩放
        QK_block = tl.dot(Q_block, K_T_block)
        QK_block = QK_block * softmax_scale
        if IS_CAUSAL:
            causal_mask = offs_q[:, None] >= curr_kv[None, :]
            QK_block = tl.where(causal_mask, QK_block, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
        QK_block = QK_block - m_ij[:, None]
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # V 路径与 K 对称
        if BITS == 8:
            v_ptrs = V_q_or_packed + v_base + curr_kv[:, None] * stride_V_s + offs_d[None, :] * stride_V_d_or_packed
            v_q = tl.load(v_ptrs, mask=(curr_kv[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM), other=0).to(tl.int8)
            V_block = (v_q.to(tl.float32) * sv_val[:, None]).to(tl.float16)
        else:
            v_pack_col = offs_d // 2
            v_is_low = (offs_d % 2) == 0
            v_pack_ptrs = V_q_or_packed + v_base + curr_kv[:, None] * stride_V_s + v_pack_col[None, :] * stride_V_d_or_packed
            v_packed_u8 = tl.load(v_pack_ptrs, mask=(curr_kv[:, None] < SEQ_LEN) & (v_pack_col[None, :] < (HEAD_DIM // 2)), other=0).to(tl.uint8)
            v_low_n = v_packed_u8 & 0xF
            v_high_n = (v_packed_u8 >> 4) & 0xF
            v_n = tl.where(v_is_low[None, :], v_low_n, v_high_n).to(tl.int16)
            v_n_signed = tl.where(v_n < 8, v_n, v_n - 16).to(tl.float32)
            V_block = (v_n_signed * sv_val[:, None]).to(tl.float16)

        vfp16_ptrs = V_FP16_OPT + vfp16_base + curr_kv[:, None] * stride_VFP16_s + offs_d[None, :] * stride_VFP16_d
        V_block_fp16 = tl.load(
            vfp16_ptrs,
            mask=(curr_kv[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM) & use_fp16_row[:, None],
            other=0.0,
        )
        V_block = tl.where(use_fp16_row[:, None], V_block_fp16, V_block)

        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)
        m_i = m_ij

    O_block = O_block / l_i[:, None]
    o_ptrs = O + o_base + offs_q[:, None] * stride_O_s + offs_d[None, :] * stride_O_d
    tl.store(o_ptrs, O_block.to(O.type.element_ty), mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM))


class FlashAttentionQuantizedForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K_q_or_packed: torch.Tensor,
        V_q_or_packed: torch.Tensor,
        scale_k: torch.Tensor,
        scale_v: torch.Tensor,
        kv_is_fp16_mask: Optional[torch.Tensor],
        K_fp16: Optional[torch.Tensor],
        V_fp16: Optional[torch.Tensor],
        causal: bool,
        softmax_scale: float,
        bits: int,
    ):
        B, H, S, D = Q.shape
        O = torch.empty_like(Q)
        assert bits in (4, 8)

        # 若无混合精度，则提供虚 mask + 虚 FP16 源（stride 为 0）
        if kv_is_fp16_mask is None:
            kv_is_fp16_mask = torch.zeros((B, H, S), device=Q.device, dtype=torch.uint8)
            K_fp16 = torch.empty(0, device=Q.device, dtype=torch.float16)
            V_fp16 = torch.empty(0, device=Q.device, dtype=torch.float16)

        def grid(meta):
            return (triton.cdiv(S, meta["BLOCK_SIZE_Q"]), B * H)

        _flash_attention_quantized_forward_with_on_the_fly_dequantization[grid](
            Q,
            K_q_or_packed,
            V_q_or_packed,
            scale_k,
            scale_v,
            K_fp16,
            V_fp16,
            kv_is_fp16_mask,
            O,
            softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K_q_or_packed.stride(0), K_q_or_packed.stride(1), K_q_or_packed.stride(2), K_q_or_packed.stride(3),
            V_q_or_packed.stride(0), V_q_or_packed.stride(1), V_q_or_packed.stride(2), V_q_or_packed.stride(3),
            # K/V 的 FP16 源 stride（如无则传 0）
            (K_fp16.stride(0) if K_fp16.numel() > 0 else 0), (K_fp16.stride(1) if K_fp16.numel() > 0 else 0), (K_fp16.stride(2) if K_fp16.numel() > 0 else 0), (K_fp16.stride(3) if K_fp16.numel() > 0 else 0),
            (V_fp16.stride(0) if V_fp16.numel() > 0 else 0), (V_fp16.stride(1) if V_fp16.numel() > 0 else 0), (V_fp16.stride(2) if V_fp16.numel() > 0 else 0), (V_fp16.stride(3) if V_fp16.numel() > 0 else 0),
            scale_k.stride(0), scale_k.stride(1), scale_k.stride(2),
            kv_is_fp16_mask.stride(0), kv_is_fp16_mask.stride(1), kv_is_fp16_mask.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B,
            NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D,
            IS_CAUSAL=1 if causal else 0,
            BITS=bits,
        )
        return O


def flash_attention_forward_with_on_the_fly_dequantization(
    Q: torch.Tensor,
    K_q_or_packed: torch.Tensor,
    V_q_or_packed: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    *,
    kv_is_fp16_mask: Optional[torch.Tensor] = None,
    K_fp16: Optional[torch.Tensor] = None,
    V_fp16: Optional[torch.Tensor] = None,
    bits: int = 8,
    causal: bool = True,
) -> torch.Tensor:
    assert Q.is_cuda and K_q_or_packed.is_cuda and V_q_or_packed.is_cuda
    D = Q.shape[-1]
    softmax_scale = 1.0 / (D ** 0.5)
    return FlashAttentionQuantizedForwardFunction.apply(
        Q, K_q_or_packed, V_q_or_packed, scale_k, scale_v,
        kv_is_fp16_mask, K_fp16, V_fp16, causal, softmax_scale, bits
    )



