# Connecting Coding Tools to Your LLM

After deploying, get your connection details:

```bash
./llm.sh creds VM_NAME
```

You'll need:
- **IP** and **Port** — `8080` for llama.cpp/Docker, `8000` for vLLM
- **API key** — generated during setup
- **Model name** — typically `qwen3.6-27b` (shown in creds output)

## Qwen Code

Config: `~/.qwen/settings.json`

```json
{
  "modelProviders": {
    "openai": [
      {
        "id": "qwen3.6-27b",
        "name": "Qwen 3.6-27B (GCP)",
        "baseUrl": "http://YOUR_VM_IP:8080/v1/",
        "description": "Qwen 3.6-27B on GCP",
        "envKey": "QWEN_GCP_KEY"
      }
    ]
  },
  "env": {
    "QWEN_GCP_KEY": "YOUR_API_KEY"
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "model": {
    "name": "qwen3.6-27b"
  }
}
```

The `id` must match what `/v1/models` returns. Use `envKey` to reference a key name defined in the `env` block (or in `~/.qwen/.env`).

To add multiple models, add more entries to the `openai` array — each with a unique `id` and its own `envKey`/`baseUrl`.

Switch models at runtime with `/model` or `qwen --model qwen3.6-27b`.

## pi.dev

Config: `~/.pi/agent/models.json` (reloads when you open `/model` — no restart needed)

```json
{
  "providers": {
    "gcp": {
      "baseUrl": "http://YOUR_VM_IP:8080/v1",
      "api": "openai-completions",
      "apiKey": "YOUR_API_KEY",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "qwen3.6-27b",
          "name": "Qwen 3.6-27B (GCP)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32000,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

The `compat` flags are important — llama-server doesn't support the `developer` role or `reasoning_effort` parameter, and uses `max_tokens` instead of `max_completion_tokens`.

The `apiKey` field accepts a literal value, an environment variable name, or a shell command prefixed with `!` (e.g., `"!cat ~/qwen-27b/.api_key"`).

Set `cost` to zero since you're self-hosting.

## OpenCode

Config: `opencode.json` in your project directory, or `~/.config/opencode/opencode.jsonc` for a global config that applies to all projects.

First, register your API key by running `/connect`, scroll to **Other**, enter `gcp` as the provider ID, and paste your API key. Credentials are stored globally in `~/.local/share/opencode/auth.json`.

Then add to your config file:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "gcp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.6-27B (GCP)",
      "options": {
        "baseURL": "http://YOUR_VM_IP:8080/v1"
      },
      "models": {
        "qwen3.6-27b": {
          "name": "Qwen 3.6-27B",
          "limit": {
            "context": 131072,
            "output": 32000
          }
        }
      }
    }
  }
}
```

Use `@ai-sdk/openai-compatible` for any server exposing `/v1/chat/completions`. The provider ID (`gcp`) must match what you entered during `/connect`. Set `limit` so OpenCode knows how much context is available.

## Continue (VS Code)

Open the config with Ctrl+Shift+P → **Continue: Open configuration file**, then add:

```yaml
models:
  - name: Qwen 3.6-27B (GCP)
    provider: openai
    model: qwen3.6-27b
    apiBase: http://YOUR_VM_IP:8080/v1
    apiKey: YOUR_API_KEY
    roles:
      - chat
      - edit
    defaultCompletionOptions:
      temperature: 0.7
      maxTokens: 4096
```

Set `roles` to control what the model is used for: `chat`, `edit`, `apply`, `autocomplete`, `embed`, or `summarize`.

## OpenAI SDK

### Python

```python
import openai

client = openai.OpenAI(
    base_url="http://YOUR_VM_IP:8080/v1",  # :8000 for vLLM
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
  baseURL: 'http://YOUR_VM_IP:8080/v1',  // :8000 for vLLM
  apiKey: 'YOUR_API_KEY',
});

const response = await client.chat.completions.create({
  model: 'qwen3.6-27b',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## Notes

- **Port**: Use `8080` for llama.cpp (direct or Docker), `8000` for vLLM.
- **Model name**: Must match what the server reports via `/v1/models`. Check with `./llm.sh creds VM_NAME` or `curl http://VM_IP:PORT/v1/models`.
- **Static IPs**: If you created the VM with `--static-ip`, the IP won't change across stop/start. Otherwise, re-check with `./llm.sh creds` after each resume.
- **All tools**: Any tool that supports OpenAI-compatible endpoints will work — just point it at `http://YOUR_VM_IP:PORT/v1` with your API key.
