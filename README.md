# quant_flash_attn

占位说明：
- 项目目标：基于 Triton 实现混合精度量化 KV Cache 的 FlashAttention，对照组（先反量化）与实验组（内核内即时反量化）两套实现与对比
- 运行入口：
  - 正确性（小输入验证即可）：`python -m quant_flash_attn.tests.test_correctness ...`
  - 基准：`python -m quant_flash_attn.tests.bench ...` 或 `scripts/bench_flash_attn_quant.sh`
- 默认参数：`B=2, H=16, S=2048, D=64, bits=8, causal=True, fp16_ratio=0.0`（8GB 友好）
- 即 
  ```
  python3 tests/bench.py --bits 4 --batch-size 2 --num-heads 16 --seq-len 4096 --head-dim 64 --fp16-ratio 0.5 --iters 10 --warmup 3 --causal
  ```
  ```
  python3 tests/bench.py --bits 8 --batch-size 2 --num-heads 16 --seq-len 4096 --head-dim 64 --fp16-ratio 0.5 --iters 10 --warmup 3 --causal
  ```

后续将补充：安装依赖、形状/位宽约束、常见问题（INT4 偶数维、mask 使用）、环境信息打印等


