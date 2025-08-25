"""
两组数据的基准测试与时延/显存对比

要点：
- `benchmark_single_implementation_latency(fn, inputs, iters, warmup)`：使用 `torch.cuda.Event` 做计时，返回均值与中位数；
- `benchmark_control_vs_experimental_memory_and_latency(args)`：
  - 生成相同输入与量化数据；
  - 分别跑对照组与实验组，记录峰值显存（max_memory_allocated）与时延；
  - 打印对比，并可选写入 CSV/JSON；
- `main_run_benchmarks(args)`：命令行入口，支持参数（B/H/S/D/bits/causal/fp16_ratio/iters/warmup）
- 每次基准前调用 `torch.cuda.reset_peak_memory_stats()`

"""

import argparse
import time
import os
import csv
from statistics import median

import torch

from quant_flash_attn.utils.quantize import (
    quantize_kv_cache_fp16_to_int8,
    quantize_kv_cache_fp16_to_packed_int4,
    make_mixed_precision_kv_mask,
)
from quant_flash_attn.ops.control_group import (
    flash_attention_forward_with_pre_dequantization,
)
from quant_flash_attn.kernels.flash_attn_quant_onthefly import (
    flash_attention_forward_with_on_the_fly_dequantization,
)


def _benchmark_once(fn, *args, **kwargs) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn(*args, **kwargs)
    # 防止懒执行
    if isinstance(out, torch.Tensor):
        _ = out.sum().item()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def benchmark_single_implementation_latency(fn, inputs, iters: int, warmup: int):
    # 预热
    for _ in range(warmup):
        _benchmark_once(fn, *inputs)
    # 记录
    times = [
        _benchmark_once(fn, *inputs)
        for _ in range(iters)
    ]
    return sum(times) / len(times), median(times)


def benchmark_control_vs_experimental_memory_and_latency(args):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    B, H, S, D = args.batch_size, args.num_heads, args.seq_len, args.head_dim
    Q = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
    K_fp16 = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
    V_fp16 = torch.randn((B, H, S, D), device="cuda", dtype=torch.float16)
    mask = make_mixed_precision_kv_mask((B, H, S), args.fp16_ratio).to(Q.device)

    if args.bits == 8:
        K_q, V_q, scale_k, scale_v = quantize_kv_cache_fp16_to_int8(K_fp16, V_fp16)
        ctrl_inputs = (Q, K_q, V_q, scale_k, scale_v)
        exp_inputs = (Q, K_q, V_q, scale_k, scale_v)
    else:
        assert D % 2 == 0
        K_qp, V_qp, scale_k, scale_v = quantize_kv_cache_fp16_to_packed_int4(K_fp16, V_fp16)
        ctrl_inputs = (Q, K_qp, V_qp, scale_k, scale_v)
        exp_inputs = (Q, K_qp, V_qp, scale_k, scale_v)

    # 对照组
    torch.cuda.reset_peak_memory_stats()
    ctrl_avg, ctrl_p50 = benchmark_single_implementation_latency(
        lambda *xs: flash_attention_forward_with_pre_dequantization(
            *xs, bits=args.bits, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=args.causal
        ),
        ctrl_inputs,
        iters=args.iters,
        warmup=args.warmup,
    )
    ctrl_mem = torch.cuda.max_memory_allocated()

    # 实验组
    torch.cuda.reset_peak_memory_stats()
    exp_avg, exp_p50 = benchmark_single_implementation_latency(
        lambda *xs: flash_attention_forward_with_on_the_fly_dequantization(
            *xs, bits=args.bits, kv_is_fp16_mask=mask, K_fp16=K_fp16, V_fp16=V_fp16, causal=args.causal
        ),
        exp_inputs,
        iters=args.iters,
        warmup=args.warmup,
    )
    exp_mem = torch.cuda.max_memory_allocated()

    report = {
        "ctrl_avg_ms": ctrl_avg,
        "ctrl_p50_ms": ctrl_p50,
        "ctrl_peak_mem": int(ctrl_mem),
        "exp_avg_ms": exp_avg,
        "exp_p50_ms": exp_p50,
        "exp_peak_mem": int(exp_mem),
    }
    print("[bench]", report)

    # 导出 CSV（可注释掉）
    csv_path = getattr(args, "csv_path", None)
    if csv_path:
        # 以追加方式写入，便于多次跑不同参数累积在同一个文件中
        header = ["bits", "batch_size", "num_heads", "seq_len", "head_dim",
                  "fp16_ratio", "causal", "iters", "warmup",
                  "ctrl_avg_ms", "ctrl_p50_ms", "ctrl_peak_mem",
                  "exp_avg_ms", "exp_p50_ms", "exp_peak_mem"]
        row = [args.bits, args.batch_size, args.num_heads, args.seq_len, args.head_dim,
               args.fp16_ratio, args.causal, args.iters, args.warmup,
               report["ctrl_avg_ms"], report["ctrl_p50_ms"], report["ctrl_peak_mem"],
               report["exp_avg_ms"], report["exp_p50_ms"], report["exp_peak_mem"]]
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(header)
            w.writerow(row)
    return report


def main_run_benchmarks():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--fp16-ratio", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    # CSV路径默认写入 tests/results.csv
    parser.add_argument("--csv-path", type=str, default="quant_flash_attn/tests/results.csv")
    args = parser.parse_args()

    benchmark_control_vs_experimental_memory_and_latency(args)


if __name__ == "__main__":
    main_run_benchmarks()



