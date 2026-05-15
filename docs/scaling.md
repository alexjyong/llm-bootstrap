# Multi-User Serving with Large Context Windows

Rough planning notes for serving Qwen 3.6-27B to many concurrent users with long context.

## The bottleneck

VRAM is split between **model weights** (fixed) and **KV cache** (scales with users x context length).

KV cache cost for Qwen 3.6-27B: ~128KB per token per user (FP16), ~64KB with FP8 KV cache.

## Quantization options (model weight size)

| Quantization | Model Weights | HF Repo | Quality vs BF16 |
|-------------|--------------|---------|-----------------|
| NVFP4 | ~14 GB | `unsloth/Qwen3.6-27B-NVFP4` | ~99% (MMLU-Pro 0.63 vs 0.64) |
| AWQ (INT4) | ~17 GB | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | ~97% |
| FP8 | ~27 GB | `Qwen/Qwen3.6-27B-FP8` | ~99.5% |
| BF16 | ~54 GB | `Qwen/Qwen3.6-27B` | baseline |

NVFP4 and AWQ leave the most VRAM for KV cache. NVFP4 has better quality retention.

## Capacity estimates (NVFP4 weights, FP8 KV cache)

With `--kv-cache-dtype fp8_e5m2` enabled (halves KV cache memory):

| GPU Setup | Total VRAM | Free for KV | Users x 32K | Users x 64K | Users x 128K |
|-----------|-----------|-------------|-------------|------------|-------------|
| 1x L4 | 24 GB | 10 GB | ~5 | ~2 | 1 |
| 2x L4 | 48 GB | 34 GB | ~17 | ~8 | ~4 |
| 1x A100 80GB | 80 GB | 66 GB | ~33 | ~16 | ~8 |
| 2x A100 80GB | 160 GB | 146 GB | ~73 | ~36 | ~18 |
| 4x A100 80GB | 320 GB | 306 GB | ~153 | ~76 | ~38 |

These are theoretical maximums. Real capacity is ~80% of these due to fragmentation, activations, etc.

## Key vLLM flags for multi-user serving

```bash
vllm serve unsloth/Qwen3.6-27B-NVFP4 \
    --tensor-parallel-size 2 \          # split across GPUs
    --kv-cache-dtype fp8_e5m2 \         # halve KV cache memory (biggest win)
    --max-model-len 131072 \            # 128K context
    --enable-prefix-caching \           # deduplicate shared system prompts
    --enable-chunked-prefill \          # overlap prefill + generation
    --max-num-seqs 32 \                 # concurrent request cap
    --gpu-memory-utilization 0.95 \
    --dtype bfloat16 \
    --trust-remote-code
```

### What each flag does

- **`--kv-cache-dtype fp8_e5m2`**: Stores KV cache in FP8 instead of FP16. Doubles user x context capacity with negligible quality loss. Single biggest lever.
- **`--enable-prefix-caching`**: If all users share the same system prompt (e.g., 2K tokens), the KV cache for that prefix is stored once instead of N times.
- **`--enable-chunked-prefill`**: Lets new requests start processing while a long prompt is still being prefilled. Improves latency for everyone.
- **`--max-num-seqs`**: Hard cap on concurrent requests. Set based on expected VRAM headroom.

## GPU hardware options on GCP

| Preset | Machine Type | GPUs | Total VRAM | Monthly cost (on-demand) |
|--------|-------------|------|-----------|------------------------|
| `l4` | g2-standard-24 | 2x L4 | 48 GB | ~$1,200 |
| `a100` | a2-highgpu-1g | 1x A100 | 80 GB | ~$3,000 |
| `a100x2` | a2-highgpu-2g | 2x A100 | 160 GB | ~$6,000 |
| (new) | a2-highgpu-4g | 4x A100 | 320 GB | ~$12,000 |

Spot instances are ~70% cheaper (as of May 2026) but can be preempted.

## MTP (Multi-Token Prediction)

Experimental speculative decoding that predicts multiple tokens per forward pass. ~30% faster generation but:
- Does NOT support parallel requests (`-np > 1`) — single-user only
- Requires a special llama.cpp branch (not vLLM)
- Not useful for multi-user serving

## Things still to figure out

- [ ] Benchmark NVFP4 vs AWQ on actual L4 hardware (quality + throughput)
- [ ] Test FP8 KV cache quality impact on long-context tasks
- [ ] Measure real vs theoretical capacity with prefix caching enabled
- [ ] Evaluate whether 4x A100 is worth it vs multiple 2x A100 instances
- [ ] Check if NVFP4 requires specific GPU architecture (Ada Lovelace / Blackwell)
- [ ] Load test to find actual max concurrent users before latency degrades
