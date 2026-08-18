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
--model <1|2|3|4|5|6>        Model selection (1=27B, 2=35B-A3B, 3=122B, 4=Gemma31B, 5=MuseGlimmer30B, 6=Qwen3.8-27B)
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

"27B Size" applies to both Qwen 3.6-27B (`--model 1`) and Qwen 3.8-27B (`--model 6`) — same dense parameter count, near-identical file sizes.

### Context targets

Qwen/Gemma/Qwen 3.8: `262k` (native), `512k`, `768k`, `1m` — anything above native enables YaRN with `--rope-scale` + `--yarn-orig-ctx` derived from the target.

Muse Glimmer (5): `131k` (native default), `262k` (YaRN 2x — the documented ceiling). Setting `--context-length` explicitly above native implies YaRN automatically.

**llama.cpp clamps the real serving context to the GGUF's own `n_ctx_train` regardless of the YaRN flags above, silently and with no error** — for every model here, not just Muse. The scripts work around it with `--override-kv <arch>.context_length=int:<N>`, where `<arch>` is that specific model's GGUF architecture tag (`MODEL_ARCH_TAGS` in the scripts: `qwen35` for models 1/6, `qwen35moe` for models 2/3, `gemma4` for model 4, `muse-glimmer` for model 5 — confirmed by reading each model's actual GGUF header, not guessed from repo names). Using the wrong key here doesn't error, it just silently doesn't take effect — this was previously hardcoded to `muse-glimmer.context_length` for every model, meaning `--context-target` above native was a no-op for every non-Muse model until this was found and fixed (see `docs/qwen-3.6-vs-3.8-research.md` for how). Muse's 13 global attention layers are NoPE (no positional encoding), which is why YaRN stretching degrades far less on that architecture than on full-RoPE models (community-verified with clean needle retrieval out to ~832K) — that verification is Muse-specific and doesn't extend to the other models.

### Muse Glimmer 30B (`--model 5`)

Muse Glimmer uses Unsloth Dynamic quants instead of the classic K-quants:

| Quant | Memory (RAM+VRAM) | Notes |
|-------|-------------------|-------|
| UD-Q2_K_XL | ~13 GB | Smallest usable |
| UD-Q3_K_XL | ~15 GB | |
| **UD-Q4_K_XL** (default) | ~18 GB | Recommended starting point |
| UD-Q6_K_XL | ~23 GB | |
| UD-Q8_K_XL | ~35 GB | Near-lossless |
| Q8_0 | ~35 GB | Classic 8-bit |

```bash
./setup_llamacpp.sh --model 5 --quant UD-Q4_K_XL --yes                  # 131k native context
./setup_llamacpp.sh --model 5 --quant UD-Q4_K_XL --context-target 262k --yes  # 262k via YaRN
```

The deploy also downloads the `mmproj-Muse-Glimmer-30B-BF16.gguf` vision adapter (multimodal input) and applies Meta's recommended sampling settings (`--temp 1.0 --top-p 0.95 --top-k 64`). Memory figures from [Unsloth's Muse Glimmer guide](https://unsloth.ai/docs/models/muse-glimmer).

Note: Muse Glimmer's chat template always emits reasoning (its effort levels are set per-request, not via `--reasoning off` — that flag is ignored by its template). Responses include `reasoning_content` alongside `content`; clients that only read `content` should budget enough `max_tokens` for the reasoning preamble.

### Qwen 3.8-27B (`--model 6`)

Newer than Qwen 3.6-27B, built on the Qwen 3.5 architecture (GGUF arch tag `qwen35`, already supported by llama.cpp — no upstream arch work needed, unlike Muse Glimmer). Same 27B dense size class, same 262K native / 1M YaRN-extensible context as the other Qwen models here, and ships with a vision encoder (`mmproj-BF16.gguf`, downloaded automatically like models 1/2/4).

```bash
./setup_llamacpp.sh --model 6 --quant Q6_K --yes
```

**⚠️ `reasoning_effort` support is incomplete until an upstream fix lands.** Qwen 3.8's chat template natively accepts a `reasoning_effort` parameter (`low`/`medium`/`high`) to tune reasoning depth per request. llama.cpp's OpenAI-compat layer currently only understands `reasoning_effort: "none"` (fully disables thinking) — any other value is silently dropped before it reaches the template. [PR #26941](https://github.com/ggml-org/llama.cpp/pull/26941) fixes this (and adds a `reasoning_strength` translation for Muse Glimmer specifically), but as of 2026-08-14 it's still open with changes requested. Until it merges — and until this repo's `setup_llamacpp.sh` picks up a llama.cpp build that includes it — only full on/off control works here (`--thinking` / `--reasoning off`); intermediate effort levels won't reach the model. **Revisit this note once #26941 (or an equivalent fix) merges upstream.**

Not eligible for `--fixed-chat-template` (that fix targets Qwen 3.5/3.6's specific template bugs and hasn't been checked against 3.8's template) or `--dflash` (no DFlash draft GGUF exists for this model).

**`--mtp` works** — unlike Qwen 3.6-27B, no separate MTP-suffixed repo is needed: the MTP head tensors ship in every quant of the default `unsloth/Qwen3.8-27B-GGUF` release already (confirmed via HF model card + GGUF tensor count, and via `common_speculative_init_result: creating MTP draft context against the target model` in llama-server's own startup log — it derives the draft context straight from the loaded target GGUF). Same caveats as model 1: forces `--parallel 1` and disables vision, both enforced automatically by `--mtp`.

Head-to-head against Qwen 3.6-27B on identical hardware (2x L4, Q6_K, same 350-token prompt, back-to-back on the same VM):

| Model | Baseline | +MTP | Speedup |
|-------|----------|------|---------|
| Qwen 3.6-27B | 10.97 tok/s | 17.94 tok/s | ~1.64x |
| Qwen 3.8-27B | 10.91 tok/s | 16.81 tok/s | ~1.54x |

Baseline decode speed is essentially identical between the two (expected — same dense 27B size class). 3.8's MTP speedup is slightly lower than 3.6's, though both are single-sample measurements on one prompt, not a rigorous benchmark. Tool-calling output (a nested-schema function call) came out byte-for-byte identical with and without `--mtp` on both models — consistent with MTP being lossless at greedy/default sampling. As of 2026-08-14 no public reports of anyone else running Qwen 3.8-27B with MTP were found, so treat these numbers as preliminary but real, working results.

**Tool-calling and agentic behavior:** both models correctly handle a nested-object function-calling schema (price ranges, category arrays, enums) via the standard `tools`/`tool_calls` API. Qwen 3.8's model card advertises "Developer Role Support... in agentic tools like Codex," but **this doesn't actually work through this repo's llama.cpp deploys** — a message with `"role": "developer"` is silently ignored by llama-server (the identical instruction sent as `"role": "system"` was followed correctly). This matches the existing caveat in [docs/client-setup.md](client-setup.md) that llama-server doesn't support the `developer` role — it's a llama-server limitation, not something specific to Qwen 3.8's template.

## Multi-GPU

Automatically detected. If you have 2+ GPUs, `llama-server` splits the model evenly across them via `--tensor-split`.
