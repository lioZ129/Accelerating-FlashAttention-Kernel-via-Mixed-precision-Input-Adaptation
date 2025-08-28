"""
对照组：单 program 内 KV 常驻调度的 FP16 FlashAttention 前向内核

要点：
- 每个 program 负责一个 Q-chunk（避免跨 program 写竞争）
- 外层循环：KV 块；在单个 program 内对每个 KV 块只加载一次并复用到该 Q-chunk 的所有行
- 在线 softmax（m_i/l_i/O_block）在单个 program 内完整归约后一次性写回
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"CHUNK_Q": 512, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"CHUNK_Q": 1024, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"CHUNK_Q": 512, "BLOCK_KV": 32}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_fp16_kv_persistent(
    Q,  # (B,H,S,D) fp16
    K,  # (B,H,S,D) fp16
    V,  # (B,H,S,D) fp16
    O,  # (B,H,S,D) fp16
    softmax_scale,
    # strides
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_d,
    stride_V_b, stride_V_h, stride_V_s, stride_V_d,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    单 program 负责一个 Q-chunk；外层循环 KV 块（在单个 program 内对每个 KV 块只加载一次并复用）。
    通过在单个 program 内完成在线 softmax 累积，避免跨 program 写竞争，保证正确性。
    """
    pid_q_chunk = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    q_chunk_start = pid_q_chunk * CHUNK_Q

    # 基址偏移
    q_base = b * stride_Q_b + h * stride_Q_h
    k_base = b * stride_K_b + h * stride_K_h
    v_base = b * stride_V_b + h * stride_V_h
    o_base = b * stride_O_b + h * stride_O_h

    offs_d = tl.arange(0, HEAD_DIM)
    offs_kv = tl.arange(0, BLOCK_KV)

    offs_q = q_chunk_start + tl.arange(0, CHUNK_Q)
    q_mask = offs_q < SEQ_LEN

    # 载入 Q-chunk
    q_ptrs = Q + q_base + offs_q[:, None] * stride_Q_s + offs_d[None, :] * stride_Q_d
    Q_block = tl.load(q_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0)

    # 在线 softmax 状态
    m_i = tl.full([CHUNK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.full([CHUNK_Q], 0.0, dtype=tl.float32)
    O_block = tl.zeros([CHUNK_Q, HEAD_DIM], dtype=tl.float32)

    # 外层循环：按 KV 块迭代（每块只加载一次）
    for start_kv in range(0, SEQ_LEN, BLOCK_KV):
        curr_kv = start_kv + offs_kv
        kv_mask = curr_kv < SEQ_LEN

        # 加载 K^T 与 V
        kT_ptrs = K + k_base + offs_d[:, None] * stride_K_d + curr_kv[None, :] * stride_K_s
        K_T_block = tl.load(
            kT_ptrs,
            mask=(offs_d[:, None] < HEAD_DIM) & kv_mask[None, :],
            other=0.0,
        )

        v_ptrs = V + v_base + curr_kv[:, None] * stride_V_s + offs_d[None, :] * stride_V_d
        V_block = tl.load(
            v_ptrs,
            mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        # 计算 QK 并缩放
        QK_block = tl.dot(Q_block, K_T_block)
        QK_block = QK_block * softmax_scale

        # 因果掩码
        if IS_CAUSAL:
            causal_mask = offs_q[:, None] >= curr_kv[None, :]
            QK_block = tl.where(causal_mask, QK_block, -1.0e6)

        # 在线 softmax 更新
        m_ij = tl.max(QK_block, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_new)
        QK_block = QK_block - m_new[:, None]
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)
        l_new = alpha * l_i + l_ij

        # 更新输出
        O_block = O_block * alpha[:, None]
        P_block = P_block.to(tl.float16)
        O_block = tl.dot(P_block, V_block, O_block)

        # 状态前移
        m_i = m_new
        l_i = l_new

    # 归一化并写回
    O_block = O_block / l_i[:, None]
    o_ptrs = O + o_base + offs_q[:, None] * stride_O_s + offs_d[None, :] * stride_O_d
    tl.store(
        o_ptrs,
        O_block.to(O.type.element_ty),
        mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


class FlashAttentionFP16ForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool, softmax_scale: float):
        assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
        B, H, S, D = Q.shape
        O = torch.empty_like(Q)

        def grid(meta):
            return (triton.cdiv(S, meta["CHUNK_Q"]), B * H)

        _flash_attention_fp16_kv_persistent[grid](
            Q, K, V, O, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D, IS_CAUSAL=1 if causal else 0,
        )
        return O


def flash_attention_fp16_kv_persistent_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Q/K/V 必须在 CUDA 上"
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape
    D = Q.shape[-1]
    softmax_scale = 1.0 / (D ** 0.5)
    return FlashAttentionFP16ForwardFunction.apply(Q, K, V, causal, softmax_scale)


# 保留旧接口以兼容现有代码
def flash_attention_fp16_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    return flash_attention_fp16_kv_persistent_forward(Q, K, V, causal)