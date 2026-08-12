# vLLM Deployment

High-throughput vLLM inference for Qwen 27B on GCP GPU VMs. vLLM supports **continuous batching** — multiple users get served simultaneously instead of queueing.

## Quick Start

```bash
# Upload and run
gcloud compute scp -r vllm/ VM_NAME:~ --zone=ZONE --project=your-project-id
gcloud compute ssh VM_NAME --zone=ZONE --project=your-project-id
chmod +x ~/vllm/setup_vllm.sh
~/vllm/setup_vllm.sh                    # interactive quant picker
~/vllm/setup_vllm.sh --quant FP8 --yes  # non-interactive

# Start
sudo systemctl start vllm-qwen-27b.service

# Get API key
cat ~/qwen-27b-vllm/.api_key
```

API is at `http://YOUR_VM_IP:8000/v1` with Bearer token auth (built-in, no nginx needed).

## Quantization Options

| Quant | VRAM | GPU Config | Quality |
|-------|------|-----------|---------|
| NVFP4 | ~14GB | 1x RTX PRO 6000 Blackwell — g4-standard-48 (full ~2.5x W4A4 speedup) or 1x L4 — g2-standard-12 (VRAM-only, no speedup) | ~99% of BF16 |
| **FP8** (default) | ~27GB | 2x L4 — g2-standard-24 | Near-lossless |
| BF16 (full) | ~54GB | 1x A100 80GB — a2-ultragpu-1g | Baseline |

**FP8** is the default — best quality-per-dollar on L4 hardware. Uses `--kv-cache-dtype fp8_e5m2` to halve KV cache memory, which is critical for fitting 64K context on 2x L4.

**NVFP4's real speedup needs Blackwell.** NVFP4 uses W4A4 (4-bit weights *and*
activations) on FP4 tensor cores, which only exist on Blackwell GPUs. On L4/A100
it falls back to vLLM's Marlin dequant path — no speedup (possibly slower than
FP8), and worse, there's an open vLLM bug
([#34694](https://github.com/vllm-project/vllm/issues/34694)) where that exact
fallback path produces garbled output on GPUs without native FP4 support
(mostly reported on consumer Blackwell, not confirmed on Ampere, but not ruled
out either). Use BF16 or FP8 on L4/A100. Use `./create_gpu_vm.sh --gpu g4` for
a single RTX PRO 6000 Blackwell GPU (96GB, actually cheaper than the A100 80GB
preset) to get NVFP4's real speedup safely — note this GPU type may need a
quota request in the GCP Console first. `setup_vllm.sh` detects the GPU at
deploy time and warns (with these specifics) if NVFP4 is selected on
non-Blackwell hardware.

## CLI Flags

```
--quant <NVFP4|FP8|BF16>  Quantization level (skip interactive picker)
--yes, -y                  Skip all prompts
--start-only               Restart existing service (after VM reboot)
--enable-tool-calling      Enable function/tool calling
--port <port>              API port (default: 8000)
```

## Managing the Service

```bash
sudo systemctl start vllm-qwen-27b.service     # Start
sudo systemctl stop vllm-qwen-27b.service      # Stop
sudo systemctl restart vllm-qwen-27b.service   # Restart
sudo systemctl status vllm-qwen-27b.service    # Status
sudo journalctl -u vllm-qwen-27b.service -f    # View logs
watch -n 1 nvidia-smi                           # Monitor GPU
```

## When to Use vLLM vs llama.cpp

| | llama.cpp | vLLM |
|---|---|---|
| **Concurrent users** | `--parallel N` slots | Continuous batching |
| **Setup** | Simple | More involved |
| **Auth** | Built-in (`--api-key`) | Built-in (`--api-key`) |
| **Quantization** | GGUF (Q3-Q8) | NVFP4, FP8, BF16 |
| **Thinking control** | Native CLI flag | N/A |
| **Best for** | Default, single user | Multiple users, throughput |
