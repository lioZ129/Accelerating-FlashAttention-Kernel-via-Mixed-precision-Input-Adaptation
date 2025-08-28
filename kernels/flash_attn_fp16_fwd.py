"""
未量化 FP16 版 FlashAttention 前向内核与封装

实现要点：
- 提供 `_flash_attention_forward_inner` 与 `_flash_attention_forward` 两个 Triton kernel
- 复用参考实现的块化与 log-sum-exp 稳定化流程：
  - Q/K/V/O 的 block_ptr 构造；
  - `m_i`、`l_i` 使用 fp32 运行状态；
  - `P_block` 用 fp16 与 V 相乘，累加用 fp32；
  - causal 与 non-causal 通过 `STAGE`/掩码控制
- 提供 `FlashAttentionFP16ForwardFunction`，以及 `flash_attention_fp16_forward` 便捷封装
- `softmax_scale` 默认 `1/sqrt(D)`

"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": 256, "BLOCK_SIZE_KV": 64}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_forward(
    Q,  # (B,H,S,D) fp16
    K,  # (B,H,S,D) fp16
    V,  # (B,H,S,D) fp16
    O,  # (B,H,S,D) fp16
    softmax_scale,
    # strides (统一为B,H,S,D)
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    说明：
    - 纯 FP16 的 FlashAttention 前向，按 (BLOCK_SIZE_Q, BLOCK_SIZE_KV) 分块计算
    - 使用数值稳定的log-sum-exp，累加与对数在 fp32 中进行
    - IS_CAUSAL 为是否应用自回归掩码
    """
    pid_q_blk = tl.program_id(0)
    pid_bh = tl.program_id(1)

    index_batch = pid_bh // NUM_HEADS
    index_head = pid_bh % NUM_HEADS

    # 计算 (batch, head) 的基址偏移
    q_base = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )
    k_base = (
        index_batch.to(tl.int64) * stride_K_batch
        + index_head.to(tl.int64) * stride_K_head
    )
    v_base = (
        index_batch.to(tl.int64) * stride_V_batch
        + index_head.to(tl.int64) * stride_V_head
    )
    o_base = (
        index_batch.to(tl.int64) * stride_O_batch
        + index_head.to(tl.int64) * stride_O_head
    )

    offs_q = pid_q_blk * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    offs_d = tl.arange(0, HEAD_DIM)

    # 载入 Q 的一个块 (BLOCK_Q, D)
    q_ptrs = Q + q_base + offs_q[:, None] * stride_Q_seq + offs_d[None, :] * stride_Q_dim
    Q_block = tl.load(q_ptrs, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    # 初始化 running max/sum 以及输出累加
    m_i = tl.full([BLOCK_SIZE_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_SIZE_Q], 1.0, dtype=tl.float32)
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # 遍历所有 KV 块
    for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        curr_kv = start_kv + offs_kv
        # 载入 K^T (D, BLOCK_KV)
        kT_ptrs = K + k_base + offs_d[:, None] * stride_K_dim + curr_kv[None, :] * stride_K_seq
        K_T_block = tl.load(
            kT_ptrs,
            mask=(offs_d[:, None] < HEAD_DIM) & (curr_kv[None, :] < SEQ_LEN),
            other=0.0,
        )
        # 计算 QK 并缩放
        QK_block = tl.dot(Q_block, K_T_block)
        QK_block = QK_block * softmax_scale

        if IS_CAUSAL:
            kv_index = curr_kv[None, :]
            q_index = offs_q[:, None]
            causal_mask = q_index >= kv_index
            QK_block = tl.where(causal_mask, QK_block, -1.0e6)

        # log-sum-exp 稳定化
        m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
        QK_block = QK_block - m_ij[:, None]
        P_block = tl.math.exp(QK_block)
        l_ij = tl.sum(P_block, axis=1)

        # 累加 O；先计算 alpha 修正项
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # 载入 V (BLOCK_KV, D)
        v_ptrs = V + v_base + curr_kv[:, None] * stride_V_seq + offs_d[None, :] * stride_V_dim
        V_block = tl.load(
            v_ptrs,
            mask=(curr_kv[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        # 累加到 O_block，P_block 转 fp16 以减少寄存器占用
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

    # 归一化并写回
    O_block = O_block / l_i[:, None]
    o_ptrs = O + o_base + offs_q[:, None] * stride_O_seq + offs_d[None, :] * stride_O_dim
    tl.store(
        o_ptrs,
        O_block.to(O.type.element_ty),
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM),
    )


class FlashAttentionFP16ForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool, softmax_scale: float):
        assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
        B, H, S, D = Q.shape
        O = torch.empty_like(Q)

        def grid(meta):
            return (triton.cdiv(S, meta["BLOCK_SIZE_Q"]), B * H)

        _flash_attention_forward[grid](
            Q,
            K,
            V,
            O,
            softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B,
            NUM_HEADS=H, 
            SEQ_LEN=S, 
            HEAD_DIM=D,
            IS_CAUSAL=1 if causal else 0,
        )
        return O


def flash_attention_fp16_forward(
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



