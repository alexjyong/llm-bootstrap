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
