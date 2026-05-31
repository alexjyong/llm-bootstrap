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

```bash
./create_gpu_vm.sh                         # interactive picker
./create_gpu_vm.sh --gpu a100              # 1x A100 40GB
./create_gpu_vm.sh --gpu a100-80           # 1x A100 80GB
./create_gpu_vm.sh --gpu a100 --static-ip  # permanent IP address
./create_gpu_vm.sh --gpu l4 --spot         # spot pricing (cheaper, can be preempted)
```

VMs auto-stop after **4 hours** by default. Override with `--auto-stop 12h` or `--no-auto-stop`.

## Managing VMs

```bash
./llm.sh list                  # list all VMs
./llm.sh creds VM_NAME        # IP, port, API key, model ID
./llm.sh test VM_NAME         # run health, auth, and inference tests
./llm.sh logs VM_NAME         # server logs
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
| vLLM | NVFP4, FP8, BF16 | **FP8** (near-lossless, fits on 2x L4) |

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

### Other

- **Manual GCP setup** (raw gcloud commands, quota checking): [docs/gcp-manual.md](docs/gcp-manual.md)
- **Docker details**: [docs/docker.md](docs/docker.md)

