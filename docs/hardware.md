# Hardware Requirements and Client Setup

## Hardware options

### Qwen 3.6-27B dense

| Quant | VRAM Needed | GCP Machine Type | GPUs |
|-------|-------------|------------------|------|
| Q3_K_M | ~14 GB | g2-standard-24 | 1x L4 24GB |
| Q4_K_M | ~17 GB | g2-standard-24 | 1x L4 24GB |
| Q6_K | ~23 GB | g2-standard-24 | 1x L4 24GB |
| Q8_0 | ~29 GB | a2-highgpu-1g | 1x A100 40GB |
| FP16 | ~54 GB | a2-ultragpu-1g | 1x A100 80GB |

Default is Q8_0. Use `--quant Q4_K_M` for smaller GPUs, or `--quant FP16` for best quality.

### Qwen 3.6-35B-A3B (GGUF only)

| Quant | VRAM Needed | GCP Machine Type | GPUs |
|-------|-------------|------------------|------|
| Q3_K_M | ~18 GB | g2-standard-24 | 1x L4 24GB |
| Q4_K_M | ~24 GB | a2-highgpu-1g | 1x A100 40GB |
| Q8_0 | ~38 GB | a2-highgpu-1g | 1x A100 40GB |
| FP16 | ~70 GB | a2-ultragpu-1g | 1x A100 80GB |

### Qwen 3.5-122B-A10B (GGUF only)

| Quant | VRAM Needed | GCP Machine Type | GPUs |
|-------|-------------|------------------|------|
| Q3_K_M | ~62 GB | a2-ultragpu-1g | 1x A100 80GB |
| Q4_K_M | ~85 GB | a2-ultragpu-2g | 2x A100 80GB |
| Q8_0 | ~140 GB | a2-ultragpu-2g | 2x A100 80GB |
| FP16 | ~244 GB | a2-ultragpu-4g | 4x A100 80GB |

## Context extension and KV cache

### KV cache presets

The KV cache stores attention state for every token in the context window. Quantizing it reduces VRAM per token, letting you fit more context or more parallel slots. Both `setup_llamacpp.sh` and `docker/setup_docker.sh` support `--kv-cache <preset>`:

| Preset | K type | V type | ~Bytes/token | Quality impact |
|--------|--------|--------|--------------|----------------|
| `q8_0` (default) | q8_0 | q8_0 | 30 | Negligible vs FP16 |
| `mixed` | q8_0 | q4_0 | 22 | Minor — keys stay high-precision, values compressed |
| `q4_0` | q4_0 | q4_0 | 15 | Moderate at long context, fine for short |

Practical impact on an A100 80GB running Qwen 3.6-27B Q6_K (~23 GB model):

| Preset | Available for KV | Max context (auto-sized) |
|--------|-----------------|-------------------------|
| `q8_0` | ~55 GB | ~196K tokens |
| `mixed` | ~55 GB | ~268K tokens |
| `q4_0` | ~55 GB | ~393K tokens |

For most workloads (chat, code review, PR agent), `q8_0` is the right default. Switch to `mixed` or `q4_0` when you need to fit more context or more parallel slots on the same GPU.

### YaRN context extension

Qwen 3.6 27B has a 262K native context window. YaRN (Yet another RoPE extensioN) scales this up to ~1M tokens by modifying the positional encoding at inference time. Both setup scripts support `--context-target <target>`:

| Target | Max tokens | YaRN | Quality |
|--------|-----------|------|---------|
| `262k` (default) | 262,144 | off | Full quality — within training distribution |
| `512k` | 524,288 | on | Modest degradation — retrieval tasks hold up well |
| `768k` | 786,432 | on | Noticeable — model may miss details at context edges |
| `1m` | 1,048,576 | on | Significant — use for search/retrieval, not precise reasoning |

Quality degrades because the model was trained on 262K context. YaRN extrapolates positional encodings beyond what the model saw during training. Tasks that require finding specific information in a large context ("needle in haystack") hold up better than tasks requiring precise reasoning over the full window.

VRAM cost: bigger context = more KV cache. Combining `--kv-cache q4_0` with `--context-target 512k` is a practical combination that fits on a single A100 80GB. Pushing to 1M tokens requires 2x A100 or aggressive KV cache quantization.

Example:
```bash
# Extended context on A100, balanced quality/capacity
./setup_llamacpp.sh --model 1 --quant Q6_K --kv-cache mixed --context-target 512k --yes

# Max context on 2x A100, aggressive compression
./setup_llamacpp.sh --model 1 --quant Q6_K --kv-cache q4_0 --context-target 1m --yes
```

### Mac (Apple Silicon)

All models run on unified memory. Rough RAM requirements are similar to the VRAM numbers above, but Apple Silicon can use swap for larger quants (with a performance hit). 32 GB Macs can comfortably run Q4_K_M of the 27B model; 64 GB+ recommended for Q8_0.

## Connecting coding tools

See [client-setup.md](client-setup.md) for copy-pasteable configs for Qwen Code, pi.dev, OpenCode, Plandex, Continue, and the OpenAI SDK.

## Using with OpenAI SDK

### Python

```python
import openai

client = openai.OpenAI(
    base_url="http://YOUR_VM_IP:8080/v1",  # llama.cpp (or :8000 for vLLM)
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Write a prime checker in Python."}],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://YOUR_VM_IP:8080/v1',  // llama.cpp (or :8000 for vLLM)
  apiKey: 'YOUR_API_KEY',
});

const response = await client.chat.completions.create({
  model: 'qwen3.6-27b',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```
