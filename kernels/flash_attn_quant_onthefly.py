"""
实验组：KV 常驻调度 + 即时反量化实现

要点：
- 每个 program 负责一个 KV 块
- KV 块反量化后在该 program 生命周期内常驻
- 循环处理所有相关的 Q 子块，每个 KV 块只反量化一次
- 分离 INT8/INT4 两套独立内核，各自 autotune 优化
- 与对照组使用相同的调度策略，公平对比反量化方式差异
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"SUB_Q": 128, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"SUB_Q": 256, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"SUB_Q": 128, "BLOCK_KV": 32}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_int8_kv_persistent(
    Q,  # (B,H,S,D) fp16
    K_q,  # (B,H,S,D) int8
    V_q,  # (B,H,S,D) int8
    SCALE_K,  # (B,H,S) fp16/fp32
    SCALE_V,  # (B,H,S) fp16/fp32
    K_FP16_OPT,  # 可选 (B,H,S,D) fp16 源
    V_FP16_OPT,  # 可选 (B,H,S,D) fp16 源
    MASK_OPT,  # (B,H,S) uint8，1 表示该行使用 FP16
    O,  # 输出 (B,H,S,D) fp16
    L,  # (B,H,S) fp32 - 存储每行的 logsumexp
    M,  # (B,H,S) fp32 - 存储每行的 max
    softmax_scale,
    # strides
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_d,
    stride_V_b, stride_V_h, stride_V_s, stride_V_d,
    stride_KFP16_b, stride_KFP16_h, stride_KFP16_s, stride_KFP16_d,
    stride_VFP16_b, stride_VFP16_h, stride_VFP16_s, stride_VFP16_d,
    stride_SC_b, stride_SC_h, stride_SC_s,
    stride_MS_b, stride_MS_h, stride_MS_s,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    stride_L_b, stride_L_h, stride_L_s,
    stride_M_b, stride_M_h, stride_M_s,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SUB_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    真正的 KV 常驻调度的 INT8 即时反量化内核：
    - 每个 program 负责一个 (batch, head, kv_block_idx)
    - 该 program 反量化一个 KV 块，然后循环处理所有相关的 Q 子块
    - KV 块只反量化一次，在该 program 生命周期内常驻
    """
    pid_kv_blk = tl.program_id(0)  # KV 块索引
    pid_bh = tl.program_id(1)      # (batch, head) 索引
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    # 当前 KV 块的全局起始位置
    kv_start = pid_kv_blk * BLOCK_KV
    
    # 基址偏移
    q_base = b * stride_Q_b + h * stride_Q_h
    k_base = b * stride_K_b + h * stride_K_h
    v_base = b * stride_V_b + h * stride_V_h
    o_base = b * stride_O_b + h * stride_O_h
    l_base = b * stride_L_b + h * stride_L_h
    m_base = b * stride_M_b + h * stride_M_h
    kfp16_base = b * stride_KFP16_b + h * stride_KFP16_h
    vfp16_base = b * stride_VFP16_b + h * stride_VFP16_h
    sk_base = b * stride_SC_b + h * stride_SC_h
    sv_base = b * stride_SC_b + h * stride_SC_h
    ms_base = b * stride_MS_b + h * stride_MS_h

    offs_d = tl.arange(0, HEAD_DIM)
    offs_kv = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = offs_kv < SEQ_LEN
    
    # 读取当前 KV 块的 scale 和 mask
    sk_ptrs = SCALE_K + sk_base + offs_kv * stride_SC_s
    sv_ptrs = SCALE_V + sv_base + offs_kv * stride_SC_s
    ms_ptrs = MASK_OPT + ms_base + offs_kv * stride_MS_s
    
    sk_val = tl.load(sk_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
    sv_val = tl.load(sv_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
    use_fp16_row = tl.load(ms_ptrs, mask=kv_mask, other=0).to(tl.int32) > 0
    
    # 反量化当前 KV 块
    # K^T 块 (D, BLOCK_KV)
    kT_ptrs = K_q + k_base + offs_d[:, None] * stride_K_d + offs_kv[None, :] * stride_K_s
    k_val_q = tl.load(
        kT_ptrs,
        mask=(offs_d[:, None] < HEAD_DIM) & kv_mask[None, :] & (~use_fp16_row[None, :]),
        other=0,
    ).to(tl.int8)
    K_T_block = (k_val_q.to(tl.float32) * sk_val[None, :]).to(tl.float16)
    
    # 覆盖 FP16 行
    kfp16_ptrs = K_FP16_OPT + kfp16_base + offs_kv[None, :] * stride_KFP16_s + offs_d[:, None] * stride_KFP16_d
    K_T_block_fp16 = tl.load(
        kfp16_ptrs,
        mask=kv_mask[None, :] & (offs_d[:, None] < HEAD_DIM) & use_fp16_row[None, :],
        other=0.0,
    )
    K_T_block = tl.where(use_fp16_row[None, :], K_T_block_fp16, K_T_block)
    
    # V 块 (BLOCK_KV, D)
    v_ptrs = V_q + v_base + offs_kv[:, None] * stride_V_s + offs_d[None, :] * stride_V_d
    v_q = tl.load(
        v_ptrs,
        mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM) & (~use_fp16_row[:, None]),
        other=0,
    ).to(tl.int8)
    V_block = (v_q.to(tl.float32) * sv_val[:, None]).to(tl.float16)
    
    # 覆盖 FP16 行
    vfp16_ptrs = V_FP16_OPT + vfp16_base + offs_kv[:, None] * stride_VFP16_s + offs_d[None, :] * stride_VFP16_d
    V_block_fp16 = tl.load(
        vfp16_ptrs,
        mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM) & use_fp16_row[:, None],
        other=0.0,
    )
    V_block = tl.where(use_fp16_row[:, None], V_block_fp16, V_block)
    
    # 循环处理所有相关的 Q 子块
    for q_start in range(0, SEQ_LEN, SUB_Q):
        offs_q = q_start + tl.arange(0, SUB_Q)
        q_mask = offs_q < SEQ_LEN
        
        # 因果掩码检查是否需要处理当前 Q 子块
        should_process = True
        if IS_CAUSAL:
            max_q_pos = tl.max(tl.where(q_mask, offs_q, -1))
            min_kv_pos = kv_start
            should_process = max_q_pos >= min_kv_pos
        
        if should_process:
            # 加载当前 Q 子块
            q_ptrs = Q + q_base + offs_q[:, None] * stride_Q_s + offs_d[None, :] * stride_Q_d
            Q_sub_block = tl.load(
                q_ptrs,
                mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0
            )
            
            # 读取当前 Q 子块的运行状态
            m_ptrs = M + m_base + offs_q * stride_M_s
            l_ptrs = L + l_base + offs_q * stride_L_s
            o_ptrs = O + o_base + offs_q[:, None] * stride_O_s + offs_d[None, :] * stride_O_d
            
            m_i = tl.load(m_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)
            l_i = tl.load(l_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            O_i = tl.load(o_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
            
            # 计算 QK
            QK_block = tl.dot(Q_sub_block, K_T_block)
            QK_block = QK_block * softmax_scale
            
            # 因果掩码
            if IS_CAUSAL:
                causal_mask = offs_q[:, None] >= offs_kv[None, :]
                QK_block = tl.where(causal_mask, QK_block, -1.0e6)
            
            # LSE 稳定化
            m_ij = tl.max(QK_block, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            
            # 更新注意力权重
            QK_block = QK_block - m_new[:, None]
            P_block = tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, axis=1)
            
            # 更新运行状态
            alpha = tl.math.exp(m_i - m_new)
            l_new = alpha * l_i + l_ij
            
            # 更新输出
            O_i = O_i * alpha[:, None]
            P_block = P_block.to(tl.float16)
            O_i = tl.dot(P_block, V_block, O_i)
            
            # 写回更新的状态
            tl.store(m_ptrs, m_new, mask=q_mask)
            tl.store(l_ptrs, l_new, mask=q_mask)
            tl.store(o_ptrs, O_i.to(O.type.element_ty), mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))


@triton.autotune(
    configs=[
        triton.Config({"SUB_Q": 128, "BLOCK_KV": 32}, num_warps=8, num_stages=3),
        triton.Config({"SUB_Q": 128, "BLOCK_KV": 64}, num_warps=8, num_stages=3),
        triton.Config({"SUB_Q": 256, "BLOCK_KV": 32}, num_warps=8, num_stages=3),
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attention_int4_kv_persistent(
    Q,  # (B,H,S,D) fp16
    K_q_packed,  # (B,H,S,D//2) uint8 packed
    V_q_packed,  # (B,H,S,D//2) uint8 packed
    SCALE_K,  # (B,H,S) fp16/fp32
    SCALE_V,  # (B,H,S) fp16/fp32
    K_FP16_OPT,  # 可选 (B,H,S,D) fp16 源
    V_FP16_OPT,  # 可选 (B,H,S,D) fp16 源
    MASK_OPT,  # (B,H,S) uint8，1 表示该行使用 FP16
    O,  # 输出 (B,H,S,D) fp16
    L,  # (B,H,S) fp32
    M,  # (B,H,S) fp32
    softmax_scale,
    # strides
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_packed_d,
    stride_V_b, stride_V_h, stride_V_s, stride_V_packed_d,
    stride_KFP16_b, stride_KFP16_h, stride_KFP16_s, stride_KFP16_d,
    stride_VFP16_b, stride_VFP16_h, stride_VFP16_s, stride_VFP16_d,
    stride_SC_b, stride_SC_h, stride_SC_s,
    stride_MS_b, stride_MS_h, stride_MS_s,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    stride_L_b, stride_L_h, stride_L_s,
    stride_M_b, stride_M_h, stride_M_s,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SUB_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    INT4 即时反量化内核：
    - 每个 program 负责一个 KV 块，反量化后常驻
    - 循环处理所有相关的 Q 子块
    """
    pid_kv_blk = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS

    kv_start = pid_kv_blk * BLOCK_KV
    
    # 基址偏移
    q_base = b * stride_Q_b + h * stride_Q_h
    k_base = b * stride_K_b + h * stride_K_h
    v_base = b * stride_V_b + h * stride_V_h
    o_base = b * stride_O_b + h * stride_O_h
    l_base = b * stride_L_b + h * stride_L_h
    m_base = b * stride_M_b + h * stride_M_h
    kfp16_base = b * stride_KFP16_b + h * stride_KFP16_h
    vfp16_base = b * stride_VFP16_b + h * stride_VFP16_h
    sk_base = b * stride_SC_b + h * stride_SC_h
    sv_base = b * stride_SC_b + h * stride_SC_h
    ms_base = b * stride_MS_b + h * stride_MS_h

    offs_d = tl.arange(0, HEAD_DIM)
    offs_kv = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = offs_kv < SEQ_LEN
    
    # 读取 scale 和 mask
    sk_ptrs = SCALE_K + sk_base + offs_kv * stride_SC_s
    sv_ptrs = SCALE_V + sv_base + offs_kv * stride_SC_s
    ms_ptrs = MASK_OPT + ms_base + offs_kv * stride_MS_s
    
    sk_val = tl.load(sk_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
    sv_val = tl.load(sv_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
    use_fp16_row = tl.load(ms_ptrs, mask=kv_mask, other=0).to(tl.int32) > 0

    # 反量化 K^T 块 - INT4 拆包
    pack_col = offs_d // 2
    is_low = (offs_d % 2) == 0
    k_pack_ptrs = K_q_packed + k_base + pack_col[:, None] * stride_K_packed_d + offs_kv[None, :] * stride_K_s
    packed_u8 = tl.load(
        k_pack_ptrs,
        mask=(pack_col[:, None] < (HEAD_DIM // 2)) & kv_mask[None, :] & (~use_fp16_row[None, :]),
        other=0,
    ).to(tl.uint8)
    
    low_n = packed_u8 & 0xF
    high_n = (packed_u8 >> 4) & 0xF
    n = tl.where(is_low[:, None], low_n, high_n).to(tl.int16)
    n_signed = tl.where(n < 8, n, n - 16).to(tl.float32)
    K_T_block = (n_signed * sk_val[None, :]).to(tl.float16)

    # 覆盖 FP16 行
    kfp16_ptrs = K_FP16_OPT + kfp16_base + offs_kv[None, :] * stride_KFP16_s + offs_d[:, None] * stride_KFP16_d
    K_T_block_fp16 = tl.load(
        kfp16_ptrs,
        mask=kv_mask[None, :] & (offs_d[:, None] < HEAD_DIM) & use_fp16_row[None, :],
        other=0.0,
    )
    K_T_block = tl.where(use_fp16_row[None, :], K_T_block_fp16, K_T_block)

    # 反量化 V 块 - INT4 拆包
    v_pack_col = offs_d // 2
    v_is_low = (offs_d % 2) == 0
    v_pack_ptrs = V_q_packed + v_base + offs_kv[:, None] * stride_V_s + v_pack_col[None, :] * stride_V_packed_d
    v_packed_u8 = tl.load(
        v_pack_ptrs,
        mask=kv_mask[:, None] & (v_pack_col[None, :] < (HEAD_DIM // 2)) & (~use_fp16_row[:, None]),
        other=0,
    ).to(tl.uint8)
    
    v_low_n = v_packed_u8 & 0xF
    v_high_n = (v_packed_u8 >> 4) & 0xF
    v_n = tl.where(v_is_low[None, :], v_low_n, v_high_n).to(tl.int16)
    v_n_signed = tl.where(v_n < 8, v_n, v_n - 16).to(tl.float32)
    V_block = (v_n_signed * sv_val[:, None]).to(tl.float16)

    # 覆盖 FP16 行
    vfp16_ptrs = V_FP16_OPT + vfp16_base + offs_kv[:, None] * stride_VFP16_s + offs_d[None, :] * stride_VFP16_d
    V_block_fp16 = tl.load(
        vfp16_ptrs,
        mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM) & use_fp16_row[:, None],
        other=0.0,
    )
    V_block = tl.where(use_fp16_row[:, None], V_block_fp16, V_block)

    # 循环处理所有相关的 Q 子块（同 INT8 版本逻辑）
    for q_start in range(0, SEQ_LEN, SUB_Q):
        offs_q = q_start + tl.arange(0, SUB_Q)
        q_mask = offs_q < SEQ_LEN
        
        should_process = True
        if IS_CAUSAL:
            max_q_pos = tl.max(tl.where(q_mask, offs_q, -1))
            min_kv_pos = kv_start
            should_process = max_q_pos >= min_kv_pos
        
        if should_process:
            # 加载 Q 子块
            q_ptrs = Q + q_base + offs_q[:, None] * stride_Q_s + offs_d[None, :] * stride_Q_d
            Q_sub_block = tl.load(
                q_ptrs,
                mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
                other=0.0
            )
            
            # 读取运行状态
            m_ptrs = M + m_base + offs_q * stride_M_s
            l_ptrs = L + l_base + offs_q * stride_L_s
            o_ptrs = O + o_base + offs_q[:, None] * stride_O_s + offs_d[None, :] * stride_O_d
            
            m_i = tl.load(m_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)
            l_i = tl.load(l_ptrs, mask=q_mask, other=0.0).to(tl.float32)
            O_i = tl.load(o_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
            
            # 计算 QK
            QK_block = tl.dot(Q_sub_block, K_T_block)
            QK_block = QK_block * softmax_scale
            
            # 因果掩码
            if IS_CAUSAL:
                causal_mask = offs_q[:, None] >= offs_kv[None, :]
                QK_block = tl.where(causal_mask, QK_block, -1.0e6)
            
            # LSE 稳定化
            m_ij = tl.max(QK_block, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            
            QK_block = QK_block - m_new[:, None]
            P_block = tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, axis=1)
            
            alpha = tl.math.exp(m_i - m_new)
            l_new = alpha * l_i + l_ij
            
            # 更新输出
            O_i = O_i * alpha[:, None]
            P_block = P_block.to(tl.float16)
            O_i = tl.dot(P_block, V_block, O_i)
            
            # 写回状态
            tl.store(m_ptrs, m_new, mask=q_mask)
            tl.store(l_ptrs, l_new, mask=q_mask)
            tl.store(o_ptrs, O_i.to(O.type.element_ty), mask=q_mask[:, None] & (offs_d[None, :] < HEAD_DIM))


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
        O = torch.zeros_like(Q)
        L = torch.zeros((B, H, S), device=Q.device, dtype=torch.float32)
        M = torch.full((B, H, S), -float("inf"), device=Q.device, dtype=torch.float32)
        assert bits in (4, 8)

        # 若无混合精度，则提供虚 mask + 虚 FP16 源
        if kv_is_fp16_mask is None:
            kv_is_fp16_mask = torch.zeros((B, H, S), device=Q.device, dtype=torch.uint8)
            K_fp16 = torch.empty(0, device=Q.device, dtype=torch.float16)
            V_fp16 = torch.empty(0, device=Q.device, dtype=torch.float16)

        def grid(meta):
            return (triton.cdiv(S, meta["BLOCK_KV"]), B * H)

        if bits == 8:
            _flash_attention_int8_kv_persistent[grid](
                Q, K_q_or_packed, V_q_or_packed, scale_k, scale_v,
                K_fp16, V_fp16, kv_is_fp16_mask, O, L, M, softmax_scale,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K_q_or_packed.stride(0), K_q_or_packed.stride(1), K_q_or_packed.stride(2), K_q_or_packed.stride(3),
                V_q_or_packed.stride(0), V_q_or_packed.stride(1), V_q_or_packed.stride(2), V_q_or_packed.stride(3),
                (K_fp16.stride(0) if K_fp16.numel() > 0 else 0), (K_fp16.stride(1) if K_fp16.numel() > 0 else 0), 
                (K_fp16.stride(2) if K_fp16.numel() > 0 else 0), (K_fp16.stride(3) if K_fp16.numel() > 0 else 0),
                (V_fp16.stride(0) if V_fp16.numel() > 0 else 0), (V_fp16.stride(1) if V_fp16.numel() > 0 else 0), 
                (V_fp16.stride(2) if V_fp16.numel() > 0 else 0), (V_fp16.stride(3) if V_fp16.numel() > 0 else 0),
                scale_k.stride(0), scale_k.stride(1), scale_k.stride(2),
                kv_is_fp16_mask.stride(0), kv_is_fp16_mask.stride(1), kv_is_fp16_mask.stride(2),
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                M.stride(0), M.stride(1), M.stride(2),
                B, NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D, IS_CAUSAL=1 if causal else 0,
            )
        else:  # bits == 4
            _flash_attention_int4_kv_persistent[grid](
                Q, K_q_or_packed, V_q_or_packed, scale_k, scale_v,
                K_fp16, V_fp16, kv_is_fp16_mask, O, L, M, softmax_scale,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K_q_or_packed.stride(0), K_q_or_packed.stride(1), K_q_or_packed.stride(2), K_q_or_packed.stride(3),
                V_q_or_packed.stride(0), V_q_or_packed.stride(1), V_q_or_packed.stride(2), V_q_or_packed.stride(3),
                (K_fp16.stride(0) if K_fp16.numel() > 0 else 0), (K_fp16.stride(1) if K_fp16.numel() > 0 else 0), 
                (K_fp16.stride(2) if K_fp16.numel() > 0 else 0), (K_fp16.stride(3) if K_fp16.numel() > 0 else 0),
                (V_fp16.stride(0) if V_fp16.numel() > 0 else 0), (V_fp16.stride(1) if V_fp16.numel() > 0 else 0), 
                (V_fp16.stride(2) if V_fp16.numel() > 0 else 0), (V_fp16.stride(3) if V_fp16.numel() > 0 else 0),
                scale_k.stride(0), scale_k.stride(1), scale_k.stride(2),
                kv_is_fp16_mask.stride(0), kv_is_fp16_mask.stride(1), kv_is_fp16_mask.stride(2),
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                M.stride(0), M.stride(1), M.stride(2),
                B, NUM_HEADS=H, SEQ_LEN=S, HEAD_DIM=D, IS_CAUSAL=1 if causal else 0,
            )
        
        # 最终归一化
        O = (O / L.unsqueeze(-1)).to(Q.dtype)
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
    assert bits in (4, 8), f"不支持的位宽: {bits}"
    if bits == 4:
        assert Q.shape[-1] % 2 == 0, "INT4 量化要求 HEAD_DIM 为偶数"
    
    D = Q.shape[-1]
    softmax_scale = 1.0 / (D ** 0.5)
    return FlashAttentionQuantizedForwardFunction.apply(
        Q, K_q_or_packed, V_q_or_packed, scale_k, scale_v,
        kv_is_fp16_mask, K_fp16, V_fp16, causal, softmax_scale, bits
    )