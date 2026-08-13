# Docker Deployment

Pre-built llama-server Docker image — no CUDA compilation on the VM. Pull the image, download a model, start the container.

## Quick Start

```bash
# From your Codespace:
./llm.sh deploy VM_NAME --backend llamacpp-docker --model 1 --quant Q6_K --yes

# Or on the VM directly:
cd ~/docker && ./setup_docker.sh --model 1 --quant Q6_K --yes
```

Available models: 1 = Qwen 3.6-27B, 2 = Qwen 3.6-35B-A3B, 3 = Gemma 4 31B, 4 = Muse Glimmer 30B. Muse Glimmer uses Unsloth Dynamic quants (`UD-Q2_K_XL` … `UD-Q8_K_XL`, plus `Q8_0`), defaults to `UD-Q4_K_XL` (~18 GB), downloads its vision adapter, and applies Meta's recommended sampling settings (temp 1.0, top-p 0.95, top-k 64) — see [docs/llamacpp.md](llamacpp.md#muse-glimmer-30b---model-5). Context targets for Muse are `131k` (native default) and `262k` (YaRN 2x, the documented ceiling). If the pre-built image predates llama.cpp's Muse Glimmer support, the script rebuilds the image locally from current master automatically.

## Why Docker?

| Problem with bash setup | Docker fix |
|------------------------|-----------|
| CUDA build takes 15-20 min per VM | Pre-built image, 30s pull |
| CUDA version mismatches | Pinned in image |
| pip/hf CLI breaking changes | Pinned in image |
| systemd service management | `docker compose up -d` |

## Manual Setup

```bash
# 1. Download model
mkdir -p ~/llama-docker/models
hf download lmstudio-community/Qwen3.6-27B-GGUF Qwen3.6-27B-Q6_K.gguf --local-dir ~/llama-docker/models
hf download lmstudio-community/Qwen3.6-27B-GGUF mmproj-Qwen3.6-27B-BF16.gguf --local-dir ~/llama-docker/models

# 2. Generate API key
openssl rand -hex 32 > ~/llama-docker/.api_key

# 3. Create .env
cat > ~/llama-docker/.env << EOF
API_KEY=$(cat ~/llama-docker/.api_key)
MODELS_DIR=$HOME/llama-docker/models
MODEL_FILE=Qwen3.6-27B-Q6_K.gguf
MMPROJ_FILE=mmproj-Qwen3.6-27B-BF16.gguf
MODEL_ALIAS=qwen3.6-27b
PORT=8080
CONTEXT_LENGTH=196608
PARALLEL=3
EOF

# 4. Start
cd ~/llama-docker
docker compose up -d
```

## DFlash Speculative Decoding

Same feature and caveats as the source-build backend (see [llamacpp.md](llamacpp.md#dflash-speculative-decoding) — forces f16 KV cache, single-GPU only, Qwen 3.6-27B only). Enable with:

```bash
./setup_docker.sh --model 1 --quant Q6_K --dflash --yes
```

The draft model is self-converted from the primary source ([z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)), not downloaded pre-built — see `docs/dflash-research.md`. Since this backend runs from a prebuilt image rather than a local build, `--dflash` does a one-time, throwaway shallow clone of llama.cpp purely to get its conversion scripts (`convert_hf_to_gguf.py`) — the server itself still runs from the prebuilt `ghcr.io` image, this clone is never built or used for anything else. The converted draft GGUF lands in the same `models/` directory that's already bind-mounted into the container, so no `docker-compose.yml` changes are needed.

## Managing the Container

```bash
cd ~/llama-docker
docker compose logs -f         # View logs
docker compose down            # Stop
docker compose up -d           # Start
docker compose pull && docker compose up -d  # Update image
```

## Building the Image Locally

If the pre-built image isn't available:

```bash
cd docker/
docker build -t ghcr.io/alexjyong/llm-bootstrap/llama-server:latest .
```

This runs the CUDA build inside Docker (~15 min). The resulting image is ~2GB (runtime only, no build tools).

## GitHub Actions CI

The workflow at `.github/workflows/docker-build.yml` builds and pushes the image weekly. Trigger a manual build:

```bash
gh workflow run docker-build.yml
```

Or with a specific llama.cpp version:

```bash
gh workflow run docker-build.yml -f llama_version=b9082
```

## Image Details

- **Base**: `nvidia/cuda:12.9.0-runtime-ubuntu22.04`
- **Size**: ~2GB (runtime only)
- **Registry**: `ghcr.io/alexjyong/llm-bootstrap/llama-server:latest`
- **Tags**: `latest` + date-stamped (e.g. `20260509`)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (required) | Bearer token for auth |
| `MODEL_FILE` | `Qwen3.6-27B-Q6_K.gguf` | GGUF filename in models dir |
| `MMPROJ_FILE` | `mmproj-Qwen3.6-27B-BF16.gguf` | Vision adapter filename |
| `MODEL_ALIAS` | `qwen3.6-27b` | Model ID for API requests |
| `PORT` | `8080` | API port |
| `CONTEXT_LENGTH` | `196608` | Total context window |
| `PARALLEL` | `3` | Concurrent request slots |
| `MODELS_DIR` | `./models` | Path to GGUF files |
