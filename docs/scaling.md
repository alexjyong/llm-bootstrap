# Multi-User Serving with Large Context Windows

Rough planning notes for serving Qwen 3.6-27B to many concurrent users with long context.

## The bottleneck

VRAM is split between **model weights** (fixed) and **KV cache** (scales with users x context length).

KV cache cost for Qwen 3.6-27B: ~128KB per token per user (FP16), ~64KB with FP8 KV cache.

## Quantization options (model weight size)

| Quantization | Model Weights | HF Repo | Quality vs BF16 |
|-------------|--------------|---------|-----------------|
| NVFP4 | ~14 GB | `unsloth/Qwen3.6-27B-NVFP4` | ~99% (MMLU-Pro 0.63 vs 0.64) |
| FP8 | ~27 GB | `Qwen/Qwen3.6-27B-FP8` | ~99.5% |
| BF16 | ~54 GB | `Qwen/Qwen3.6-27B` | baseline |

NVFP4 leaves the most VRAM for KV cache with good quality retention. Note the
"real" NVFP4 story is more nuanced than the weight-size table above suggests —
see the GPU architecture callout below.

## Capacity estimates (NVFP4 weights, FP8 KV cache)

With `--kv-cache-dtype fp8_e5m2` enabled (halves KV cache memory):

| GPU Setup | Total VRAM | Free for KV | Users x 32K | Users x 64K | Users x 128K |
|-----------|-----------|-------------|-------------|------------|-------------|
| 1x L4 | 24 GB | 10 GB | ~5 | ~2 | 1 |
| 2x L4 | 48 GB | 34 GB | ~17 | ~8 | ~4 |
| 1x A100 40GB | 40 GB | 26 GB | ~13 | ~6 | ~3 |
| 2x A100 40GB | 80 GB | 66 GB | ~33 | ~16 | ~8 |
| 4x A100 40GB | 160 GB | 146 GB | ~73 | ~36 | ~18 |

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
| `a100` | a2-highgpu-1g | 1x A100 | 40 GB | ~$3,000 |
| `a100-80` | a2-ultragpu-1g | 1x A100 (80GB) | 80 GB | ~$3,700 |
| `a100x2` | a2-highgpu-2g | 2x A100 | 80 GB | ~$6,000 |
| (new) | a2-highgpu-4g | 4x A100 | 160 GB | ~$12,000 |
| `g4` | g4-standard-48 | 1x RTX PRO 6000 Blackwell | 96 GB | ~$3,285 |
| `g4-mini` | g4-standard-12 | 1/4 RTX PRO 6000 Blackwell (MIG) | ~24 GB | cheap smoke test |

Spot instances are ~70% cheaper (as of May 2026) but can be preempted.

### NVFP4 needs Blackwell for its real speedup

NVFP4's headline win (Unsloth's ~2.5x generation speedup on Qwen3.6-27B) comes
from W4A4 — 4-bit weights *and* activations run directly on FP4 tensor cores.
Those tensor cores only exist on **Blackwell** GPUs (RTX 50-series, RTX PRO
6000 Blackwell, B200, B300). L4 (Ada Lovelace) and A100 (Ampere) don't have
them, so NVFP4 on those presets loads and shrinks VRAM but gets **no speedup**
— likely slower than FP8 due to dequant overhead, since it falls back to
vLLM's Marlin dequant path. Worse, there's an open, unresolved vLLM bug
([vllm-project/vllm#34694](https://github.com/vllm-project/vllm/issues/34694))
where that exact fallback path produces **garbled output** on GPUs without
native FP4 tensor cores — reports are mostly consumer Blackwell (sm_120), not
confirmed on true Ampere/A100, but A100 hits the same fallback code, so it
isn't ruled out either. Use BF16 or FP8 on L4/A100 — NVFP4 buys nothing there
and carries real correctness risk on top.

`g4` (1x RTX PRO 6000 Blackwell, 96GB) is GCP's cheapest single-GPU Blackwell
option and is actually cheaper and roomier than `a100-80` (~$3,285/mo vs.
~$3,700/mo, 96GB vs. 80GB VRAM) — it's the only preset here that can deliver
NVFP4's real speedup. It's a newer SKU, though: this project currently has
**no granted GPU quota** for `nvidia-rtx-pro-6000` (it doesn't show up in
`gcloud compute regions describe REGION --format="table(quotas.metric,...)"`
the way A100/L4 do), so request quota in the GCP Console before trying to
provision one. `g4-mini` (a quarter-GPU MIG slice, ~24GB) is a cheap way to
confirm quota/availability first — note it's zone-restricted more narrowly
than `g4`: as of this writing `g4-standard-12` is only orderable in 10 zones
(`asia-east1-a`, `europe-north1-b`, `europe-west2-c`, `europe-west4-a`,
`us-central1-b`, `us-east1-d`, `us-east5-a/b/c`, `us-south1-b`) vs. `~38` for
the full-GPU `g4-standard-48`. `create_gpu_vm.sh` checks actual machine-type
zone availability (not just raw GPU chip presence) for `g4`/`g4-mini`, so it
won't waste attempts on zones where the specific shape doesn't exist.

`vllm/setup_vllm.sh` detects GPU compute capability at deploy time and warns
(with a confirmation prompt, unless `--yes`) if NVFP4 is selected on a
non-Blackwell GPU.

## MTP (Multi-Token Prediction)

Speculative decoding using draft prediction heads built into the model. ~2x faster generation. Supports multi-GPU tensor parallelism and parallel request slots. Prompt processing speed is reduced (~50%).

## Things still to figure out

- [ ] Benchmark NVFP4 vs FP8 on actual L4 hardware (quality + throughput)
- [ ] Test FP8 KV cache quality impact on long-context tasks
- [ ] Measure real vs theoretical capacity with prefix caching enabled
- [ ] Evaluate whether 4x A100 is worth it vs multiple 2x A100 instances
- [x] Check if NVFP4 requires specific GPU architecture — yes, Blackwell (see above); request G4 quota, then benchmark real W4A4 speedup on `g4` once available
- [ ] Load test to find actual max concurrent users before latency degrades
