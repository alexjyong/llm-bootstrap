# LLM Deployment Toolkit for GCP

Deploy open-weight LLMs (Qwen 3.6-27B by default) on GCP GPU VMs with OpenAI-compatible APIs.

> **Stop VMs when not in use.** GPU billing runs 24/7 while the instance is up. Always `./llm.sh stop VM_NAME` when done — the disk is preserved and you can resume in seconds with `./llm.sh resume VM_NAME`.

## Prerequisites

### Option A: GitHub Codespace (recommended)

Fork/clone the repo and open a Codespace — `gcloud`, `git`, and Python are pre-installed. Then authenticate:

```bash
gcloud auth login
gcloud config set project your-project-id
```

### Option B: Local machine

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth login
gcloud config set project your-project-id
git clone https://github.com/your-username/llm-bootstrap.git && cd llm-bootstrap 
```

You also need `bash` 4+, `ssh`, and `python3` (for model downloads). macOS ships bash 3 — use `brew install bash` or run from Linux/WSL.

### Configuration

Set your GCP project ID and subnet name as environment variables. These will be used by the scripts to target the correct resources:

```bash
export GCP_PROJECT="your-project-id"
export GCP_SUBNET="default"         # optional, defaults to "default"
export NGROK_AUTHTOKEN="..."        # optional, only needed for --ngrok deploys
export LLM_API_KEY="..."            # optional, use a fixed API key instead of a random one
```

If you are using a custom VPC or non-default subnet, ensure `GCP_SUBNET` matches your configuration.

## Quick Start

```bash
# 1. Create a GPU VM
./create_gpu_vm.sh --gpu a100

# 2. Deploy Qwen 3.6-27B (VM name is printed after step 1)
./llm.sh deploy VM_NAME --backend llamacpp-docker --model 1 --quant Q6_K --yes

# 3. Get your API credentials
./llm.sh creds VM_NAME
```

### Test it

```bash
./llm.sh test VM_NAME
```

This checks health, model loading, auth, chat completion, and streaming — prints pass/fail for each.

### Connect your tools

Set up Qwen Code, pi.dev, OpenCode, Plandex, Continue, or the OpenAI SDK to use your deployed model: [docs/client-setup.md](docs/client-setup.md)

## Backends

**llama.cpp** (direct or Docker) is the recommended default. Use vLLM when you need high-throughput concurrent serving.

| | llama.cpp | llama.cpp (Docker) | vLLM | vLLM (Docker) |
|---|---|---|---|---|
| **Setup** | Build from source (~15 min) | Pre-built image, no compile | Install via pip | Official image, no pip install |
| **Thinking control** | Native CLI flag | Via env var | N/A | N/A |
| **Concurrent users** | `--parallel N` slots | `--parallel N` slots | Continuous batching | Continuous batching |
| **Best for** | Default, full control | Fast deploy | High throughput, many users | Fastest vLLM deploy |

All backends provide built-in OpenAI-compatible APIs with Bearer token auth.

Docs: [llama.cpp](docs/llamacpp.md) | [Docker](docs/docker.md) | [vLLM](vllm/)

## GPU Presets

| Preset | Machine | GPUs | VRAM | Best for |
|--------|---------|------|------|----------|
| `l4` (default) | g2-standard-24 | 2x L4 | 48GB | GGUF quants (Q4–Q8) |
| `a100` | a2-highgpu-1g | 1x A100 | 40GB | Fast inference, large context |
| `a100-80` | a2-ultragpu-1g | 1x A100 | 80GB | MTP, large context + headroom |
| `a100x2` | a2-highgpu-2g | 2x A100 | 80GB | Full precision, multi-user vLLM |
| `g4` | g4-standard-48 | 1x RTX PRO 6000 Blackwell | 96GB | vLLM NVFP4 with real W4A4 speedup |
| `g4-mini` | g4-standard-12 | 1/4 RTX PRO 6000 Blackwell (MIG) | ~24GB | Cheap smoke test for G4 quota |

```bash
./create_gpu_vm.sh                         # interactive picker
./create_gpu_vm.sh --gpu a100              # 1x A100 40GB
./create_gpu_vm.sh --gpu a100-80           # 1x A100 80GB
./create_gpu_vm.sh --gpu g4                # 1x RTX PRO 6000 Blackwell (NVFP4 speedup)
./create_gpu_vm.sh --gpu a100 --static-ip  # permanent IP address
./create_gpu_vm.sh --gpu l4 --spot         # spot pricing (cheaper, can be preempted)
```

`g4` may need a GPU quota request in the GCP Console first (`nvidia-rtx-pro-6000` is a newer SKU than L4/A100).

VMs auto-stop after **4 hours** by default. Override with `--auto-stop 12h` or `--no-auto-stop`.

## Managing VMs

```bash
./llm.sh list                  # list all VMs
./llm.sh creds VM_NAME        # IP, port, API key, model ID
./llm.sh test VM_NAME         # run health, auth, and inference tests
./llm.sh logs VM_NAME         # server logs (last 50 lines)
./llm.sh logs VM_NAME -f      # keep streaming new log lines (like tail -f)
./llm.sh stop VM_NAME         # stop (keeps disk, stops billing)
./llm.sh resume VM_NAME       # start + restart service (auto-detects backend)
./llm.sh ssh VM_NAME          # SSH in
./llm.sh config VM_NAME context-length 262144   # change context window
./llm.sh config VM_NAME parallel 2              # change concurrent slots
./llm.sh delete VM_NAME       # delete VM and disk (asks for confirmation)
```

### Resuming a stopped VM

The disk (model, config, API key) is preserved across stop/start — only the IP changes.

```bash
./llm.sh resume VM_NAME       # start VM + restart service
./llm.sh creds VM_NAME        # get the new IP + API key
```

Use `--static-ip` when creating the VM to keep the same IP.

## Quantization

| Engine | Options | Recommended |
|--------|---------|-------------|
| llama.cpp (GGUF) | Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 | **Q6_K** (best quality/VRAM tradeoff) |
| llama.cpp (Muse Glimmer) | UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, Q8_0 | **UD-Q4_K_XL** (~18 GB, Unsloth Dynamic) |
| vLLM | NVFP4, FP8, BF16 | **FP8** (near-lossless, fits on 2x L4) |

Muse Glimmer 30B (`--model 5`, llama.cpp backends only) is Meta's agentic vision model. Deploys include its `mmproj` vision adapter and Meta's recommended sampling settings (temp 1.0, top-p 0.95, top-k 64). Its template always reasons first (responses carry `reasoning_content`), so budget `max_tokens` accordingly. See [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF).

Higher quants = better quality but more VRAM. Hardware requirements: [docs/hardware.md](docs/hardware.md)

## Advanced

### Firewall and subnet setup

If you are using a custom VPC or a non-default subnet, you need to ensure that inbound TCP traffic is allowed on the backend ports (defaulting to 8080 for llama.cpp and 8000 for vLLM):

```bash
# Example: Allow API traffic from your IP only (recommended)
gcloud compute firewall-rules create allow-llm-api \
  --allow=tcp:8080,tcp:8000 \
  --source-ranges=YOUR_IP/32 \
  --project=your-project-id
```


Port `8080` is llama.cpp, port `8000` is vLLM. All backends require a Bearer token regardless of firewall rules.

Set `GCP_SUBNET` to use a different subnet. It must exist in the region where the VM is created — `create_gpu_vm.sh` loops through zones and skips regions where the subnet isn't available.

### Exposing via ngrok

Pass `--ngrok` to `deploy` (or answer "yes" to the wizard prompt) to tunnel the backend through [ngrok](https://ngrok.com) instead of relying on the VM's ephemeral IP:

```bash
export NGROK_AUTHTOKEN="..."   # from https://dashboard.ngrok.com/get-started/your-authtoken
./llm.sh deploy VM_NAME --backend llamacpp --ngrok --yes
```

No firewall rule is needed for the backend port — `./llm.sh creds`/`test`/`info` will show and use the `https://*.ngrok-free.app` URL once it's live. On the free tier this URL changes every time the `ngrok` service restarts (e.g. on VM reboot); use a reserved domain in your ngrok account if you need a stable URL.

### Using a fixed API key

By default every deploy generates a new random API key (`openssl rand -hex 32`) for the backend. To reuse the same key across VMs — e.g. so you don't have to update client configs every time you redeploy — pass `--api-key` (or answer the wizard prompt), or set `LLM_API_KEY`:

```bash
export LLM_API_KEY="sk-my-fixed-key"
./llm.sh deploy VM_NAME --backend llamacpp --yes
# or: ./llm.sh deploy VM_NAME --backend llamacpp --api-key sk-my-fixed-key --yes
```

### Fixed chat template for Qwen 3.5/3.6

Pass `--fixed-chat-template` (or answer the wizard prompt) to use [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) instead of the model's default chat template. It's a community rewrite that claims to fix agentic tool-calling loops, KV-cache invalidation between turns, and llama.cpp/minijinja incompatibilities in Alibaba's official template:

```bash
./llm.sh deploy VM_NAME --backend llamacpp --model 1 --fixed-chat-template --yes
./llm.sh deploy VM_NAME --backend vllm --fixed-chat-template --yes
```

This is opt-in and only applies to Qwen models (it's rejected for the Gemma and Muse Glimmer options on the llama.cpp backends). The template is downloaded from a pinned commit, not `main`, so a running deploy won't change behavior if the upstream repo is edited later — these are third-party, unverified claims, so treat it as experimental.

### DFlash speculative decoding

Pass `--dflash` (llama.cpp backends only, model 1) for speculative decoding — roughly 3.75x faster generation than plain inference:

```bash
./llm.sh deploy VM_NAME --backend llamacpp --model 1 --dflash --yes
```

The draft model is self-converted from the primary source ([z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)) on first run rather than downloaded pre-built — every pre-built DFlash GGUF found on Hugging Face turned out to be an unofficial third-party conversion, and several popular ones silently fail to load. Mutually exclusive with `--mtp`. See [docs/llamacpp.md](docs/llamacpp.md#dflash-speculative-decoding) for caveats (forces f16 KV cache, single-GPU only) and [docs/dflash-research.md](docs/dflash-research.md) for the full research behind these tradeoffs.

### Other

- **Manual GCP setup** (raw gcloud commands, quota checking): [docs/gcp-manual.md](docs/gcp-manual.md)
- **Docker details**: [docs/docker.md](docs/docker.md)

