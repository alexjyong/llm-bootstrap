# llama.cpp Deployment

Direct llama.cpp inference — no daemon, no Python, no wrapper. Uses `llama-server` pre-built CUDA binaries from GitHub releases.

## Quick Start

```bash
# Upload and run
gcloud compute scp setup_llamacpp.sh VM_NAME:~ --zone=ZONE --project=your-project-id
gcloud compute ssh VM_NAME --zone=ZONE --project=your-project-id
chmod +x ~/setup_llamacpp.sh
~/setup_llamacpp.sh                              # interactive picker
~/setup_llamacpp.sh --model 1 --quant Q6_K --yes # non-interactive

# Start
sudo systemctl start llamacpp.service

# Get API key
cat ~/qwen-27b-llamacpp/.api_key
```

Or deploy from your Codespace:

```bash
./llm.sh deploy VM_NAME --backend llamacpp --model 1 --quant Q6_K --yes
```

API is at `http://YOUR_VM_IP:8080/v1` with Bearer token auth.

## CLI Flags

```
--model <1|2|3>              Model selection (1=27B, 2=35B-A3B, 3=122B)
--quant <Q3_K_M|...|Q8_0>   Quantization level
--yes, -y                    Skip all prompts
--start-only                 Restart existing service
--port <port>                API port (default: 8080)
--context-length <N>         Context window (default: 65536)
--thinking                   Enable thinking mode (default: disabled)
```

## Thinking Mode

Thinking is **disabled by default**. Qwen 3.6 was designed to work well without thinking, and disabling it gives faster responses without quality loss.

To enable thinking:

```bash
./setup_llamacpp.sh --model 1 --quant Q6_K --thinking --yes
```

This is handled natively by `llama-server` via `--chat-template-kwargs '{"enable_thinking":false}'`.

## DFlash Speculative Decoding

DFlash pairs a small draft model with the target model to draft multiple tokens per forward pass — roughly **3.75x faster generation** than plain inference on Qwen 3.6-27B, ahead of `--mtp`'s ~2x. Enable with:

```bash
./setup_llamacpp.sh --model 1 --quant Q6_K --dflash --yes
```

Mutually exclusive with `--mtp` (both are speculative-decoding modes; pick one).

**The draft model is self-converted, not downloaded pre-built.** Every pre-built DFlash GGUF found on Hugging Face is a third-party conversion of [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) (the original MIT-licensed release) — several popular ones predate llama.cpp's architecture rename and fail to load. Instead, `--dflash` downloads the primary-source safetensors plus just the target's tokenizer files, then converts locally via llama.cpp's own `convert_hf_to_gguf.py`. This happens once (cached in the model directory) and adds a few minutes to first-time setup, plus a one-time install of llama.cpp's Python conversion dependencies (`torch`, `transformers`, etc. — CPU-only, not the CUDA build).

Caveats, in order of how confident the underlying research is:

- **Forces f16 KV cache**, overriding whatever `--kv-cache` preset you picked. Quantized KV cache (q8_0/mixed/q4_0) measured a 7x slowdown in draft verification speed in third-party testing. This costs roughly double the VRAM per token, so auto-sized context will come out smaller than without `--dflash`.
- **Only validated single-GPU.** No source found tested `--tensor-split` multi-GPU with DFlash — it's not blocked, but expect the unexpected on the `l4` (2x L4) preset.
- Restricted to Qwen 3.6-27B (model 1) — no DFlash draft exists for the other models in this repo's registry.

See `docs/dflash-research.md` for the full research behind these decisions.

## Managing the Service

```bash
sudo systemctl start llamacpp.service      # Start
sudo systemctl stop llamacpp.service       # Stop
sudo systemctl restart llamacpp.service    # Restart
sudo systemctl status llamacpp.service     # Status
sudo journalctl -u llamacpp.service -f     # View logs
watch -n 1 nvidia-smi                      # Monitor GPU
```

## Quantization Options

| Quant | 27B Size | 35B-A3B Size | Quality |
|-------|----------|-------------|---------|
| Q3_K_M | ~14GB | ~18GB | Good |
| Q4_K_M | ~17GB | ~24GB | Better |
| Q5_K_M | ~20GB | ~28GB | Great |
| **Q6_K** (default for 27B) | ~23GB | ~32GB | Near-lossless |
| Q8_0 | ~29GB | ~38GB | Best |

## Multi-GPU

Automatically detected. If you have 2+ GPUs, `llama-server` splits the model evenly across them via `--tensor-split`.
