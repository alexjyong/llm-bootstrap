#!/bin/bash

# llama.cpp setup script for Google Cloud GPU VMs.
# Uses llama-server directly — no daemon, no Python, no nginx.
# Native thinking control, built-in API key auth, OpenAI-compatible API.
#
# Sets up:
#   - llama-server (built from source with CUDA)
#   - GGUF model download from Hugging Face
#   - Built-in API key auth (--api-key)
#   - systemd service
#
# Usage:
#   ./setup_llamacpp.sh                              # interactive model + quant picker
#   ./setup_llamacpp.sh --model 1 --quant Q6_K       # skip TUI
#   ./setup_llamacpp.sh --model 1 --quant Q6_K --yes # fully non-interactive
#   ./setup_llamacpp.sh --start-only                 # restart existing service
#   ./setup_llamacpp.sh --thinking                   # enable thinking mode

set -e

# ===================================================================
# Model registry — add new models here
# ===================================================================
MODEL_NAMES=(
    "Qwen 3.6-27B (dense)"
    "Qwen 3.6-35B-A3B (MoE)"
    "Qwen 3.5-122B-A10B (MoE)"
)
MODEL_DESCS=(
    "27B params, all active. Best quality per token."
    "35B total, 3B active per token. Fast, lower quality."
    "122B total, 10B active per token. Needs multi-GPU."
)
MODEL_HF_REPOS=(
    "lmstudio-community/Qwen3.6-27B-GGUF"
    "lmstudio-community/Qwen3.6-35B-A3B-GGUF"
    "unsloth/Qwen3.5-122B-A10B-GGUF"
)
MODEL_FILE_PATTERNS=(
    "Qwen3.6-27B"
    "Qwen3.6-35B-A3B"
    "Qwen3.5-122B-A10B"
)
MODEL_DEFAULT_QUANTS=("Q6_K" "Q4_K_M" "Q4_K_M")
MODEL_DIR_NAMES=("qwen-27b-llamacpp" "qwen-35b-llamacpp" "qwen-122b-llamacpp")
MODEL_ALIASES=("qwen3.6-27b" "qwen3.6-35b-a3b" "qwen3.5-122b-a10b")
MODEL_MMPROJ_FILES=(
    "mmproj-Qwen3.6-27B-BF16.gguf"
    "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
    ""
)

QUANT_OPTIONS=("Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0")

get_vram_estimate() {
    local model_idx=$1 quant=$2
    case "$model_idx:$quant" in
        0:Q3_K_M) echo "~14 GB" ;; 0:Q4_K_M) echo "~17 GB" ;; 0:Q5_K_M) echo "~20 GB" ;;
        0:Q6_K)   echo "~23 GB" ;; 0:Q8_0)   echo "~29 GB" ;;
        1:Q3_K_M) echo "~18 GB" ;; 1:Q4_K_M) echo "~24 GB" ;; 1:Q5_K_M) echo "~28 GB" ;;
        1:Q6_K)   echo "~32 GB" ;; 1:Q8_0)   echo "~38 GB" ;;
        2:Q3_K_M) echo "~62 GB" ;; 2:Q4_K_M) echo "~85 GB" ;; 2:Q5_K_M) echo "~100 GB" ;;
        2:Q6_K)   echo "~115 GB";; 2:Q8_0)   echo "~140 GB";;
        *) echo "unknown" ;;
    esac
}

get_gguf_filename() {
    local pattern=$1 quant=$2
    echo "${pattern}-${quant}.gguf"
}

# ===================================================================
# Parse arguments
# ===================================================================
AUTO_YES=false
START_ONLY=false
MODEL_ARG=""
QUANT=""
PORT=8080
CONTEXT_LENGTH="auto"
PARALLEL=3
ENABLE_THINKING=false
KV_CACHE_PRESET=""
CONTEXT_TARGET=""

show_usage() {
    cat << 'EOF'
llama.cpp Setup for Qwen Models

Usage: ./setup_llamacpp.sh [options]

Options:
  --model <number|name>       Model selection (1=27B, 2=35B-A3B, 3=122B)
  --quant <Q3_K_M|...|Q8_0>  Quantization level (skip interactive picker)
  --yes, -y                   Skip all prompts (non-interactive)
  --start-only                Skip installation, just start existing service
  --port <port>               API port (default: 8080)
  --context-length <N>        Exact context window in tokens (overrides --context-target)
  --context-target <target>   Context target: 262k, 512k, 768k, 1m (default: 262k)
                              Targets above 262k enable YaRN rope scaling
  --kv-cache <preset>         KV cache preset: q8_0, mixed, q4_0 (default: q8_0)
  --parallel <N>              Concurrent request slots (default: 4)
  --thinking                  Enable thinking mode (default: disabled)

Models:
  1  Qwen 3.6-27B        27B params, all active. Best quality per token.
  2  Qwen 3.6-35B-A3B    35B total, 3B active. Fast, lower quality.
  3  Qwen 3.5-122B-A10B  122B total, 10B active. Needs multi-GPU.

Quants:     Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
KV cache:   q8_0 (best quality), mixed (q8 keys/q4 values), q4_0 (max context)
Context:    262k (native), 512k/768k/1m (YaRN — quality degrades beyond native)

Examples:
  ./setup_llamacpp.sh                                # interactive picker
  ./setup_llamacpp.sh --model 1 --quant Q6_K --yes   # Qwen 27B Q6_K automated
  ./setup_llamacpp.sh --model 1 --quant Q4_K_M --yes # cheapest option
  ./setup_llamacpp.sh --start-only                   # restart after VM reboot
  ./setup_llamacpp.sh --thinking                     # enable thinking mode
  ./setup_llamacpp.sh --kv-cache mixed --context-target 512k --yes  # extended context

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --start-only)
            START_ONLY=true
            shift
            ;;
        --model)
            MODEL_ARG="$2"
            shift 2
            ;;
        --quant)
            QUANT="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --context-length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --kv-cache)
            KV_CACHE_PRESET="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --context-target)
            CONTEXT_TARGET="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --thinking)
            ENABLE_THINKING=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ===================================================================
# Handle --start-only
# ===================================================================
if [ "$START_ONLY" = "true" ]; then
    SERVICE_NAME="llamacpp"
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        echo "ERROR: Service $SERVICE_NAME not found. Run full setup first."
        exit 1
    fi
    echo "Starting $SERVICE_NAME.service..."
    sudo systemctl start "$SERVICE_NAME.service"
    echo "Service started."
    echo ""
    echo "Check status:  sudo systemctl status $SERVICE_NAME.service"
    echo "View logs:     sudo journalctl -u $SERVICE_NAME.service -f"
    exit 0
fi

# ===================================================================
# Resolve model selection
# ===================================================================
resolve_model() {
    local arg="$1"
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        local idx=$((arg - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#MODEL_NAMES[@]} ]; then
            MODEL_IDX=$idx
            return 0
        fi
        echo "ERROR: Model number $arg out of range (1-${#MODEL_NAMES[@]})"
        exit 1
    fi
    echo "ERROR: Unknown model '$arg'. Use a number (1-${#MODEL_NAMES[@]})."
    exit 1
}

if [ -n "$MODEL_ARG" ]; then
    resolve_model "$MODEL_ARG"
elif [ "$AUTO_YES" = "true" ]; then
    MODEL_IDX=0
else
    echo ""
    echo "Select model:"
    echo ""
    for i in "${!MODEL_NAMES[@]}"; do
        echo "  $((i+1))) ${MODEL_NAMES[$i]}"
        echo "     ${MODEL_DESCS[$i]}"
        echo ""
    done
    while true; do
        read -p "Model [1-${#MODEL_NAMES[@]}] (Enter for default): " choice
        if [ -z "$choice" ]; then
            MODEL_IDX=0
            break
        fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#MODEL_NAMES[@]} ]; then
            MODEL_IDX=$((choice - 1))
            break
        fi
        echo "  Invalid choice."
    done
fi

MODEL_NAME="${MODEL_NAMES[$MODEL_IDX]}"
HF_REPO="${MODEL_HF_REPOS[$MODEL_IDX]}"
FILE_PATTERN="${MODEL_FILE_PATTERNS[$MODEL_IDX]}"
DIR_NAME="${MODEL_DIR_NAMES[$MODEL_IDX]}"
MODEL_ALIAS="${MODEL_ALIASES[$MODEL_IDX]}"
MMPROJ_FILE="${MODEL_MMPROJ_FILES[$MODEL_IDX]}"
WORK_DIR="$HOME/$DIR_NAME"

# ===================================================================
# Resolve quant
# ===================================================================
if [ -z "$QUANT" ]; then
    DEFAULT_QUANT="${MODEL_DEFAULT_QUANTS[$MODEL_IDX]}"

    if [ "$AUTO_YES" = "true" ]; then
        QUANT="$DEFAULT_QUANT"
    else
        echo ""
        echo "Select quantization:"
        echo ""
        for i in "${!QUANT_OPTIONS[@]}"; do
            local_vram=$(get_vram_estimate $MODEL_IDX "${QUANT_OPTIONS[$i]}")
            default_marker=""
            if [ "${QUANT_OPTIONS[$i]}" = "$DEFAULT_QUANT" ]; then
                default_marker=" (default)"
            fi
            echo "  $((i+1))) ${QUANT_OPTIONS[$i]}  ${local_vram}${default_marker}"
        done
        echo ""
        while true; do
            read -p "Quant [1-${#QUANT_OPTIONS[@]}] (Enter for default): " choice
            if [ -z "$choice" ]; then
                QUANT="$DEFAULT_QUANT"
                break
            fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#QUANT_OPTIONS[@]} ]; then
                QUANT="${QUANT_OPTIONS[$((choice - 1))]}"
                break
            fi
            echo "  Invalid choice."
        done
    fi
fi

# ===================================================================
# Resolve KV cache preset
# ===================================================================
KV_CACHE_OPTIONS=("q8_0" "mixed" "q4_0")
KV_CACHE_LABELS=(
    "q8_0   ~30 bytes/token  best quality"
    "mixed  ~22 bytes/token  q8 keys, q4 values — minor quality loss"
    "q4_0   ~15 bytes/token  max context per GB — moderate quality loss"
)

if [ -z "$KV_CACHE_PRESET" ]; then
    if [ "$AUTO_YES" = "true" ]; then
        KV_CACHE_PRESET="q8_0"
    else
        echo ""
        echo "Select KV cache type:"
        echo ""
        for i in "${!KV_CACHE_OPTIONS[@]}"; do
            default_marker=""
            [ "$i" = "0" ] && default_marker=" (default)"
            echo "  $((i+1))) ${KV_CACHE_LABELS[$i]}${default_marker}"
        done
        echo ""
        while true; do
            read -p "KV cache [1-${#KV_CACHE_OPTIONS[@]}] (Enter for default): " choice
            if [ -z "$choice" ]; then KV_CACHE_PRESET="q8_0"; break; fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#KV_CACHE_OPTIONS[@]} ]; then
                KV_CACHE_PRESET="${KV_CACHE_OPTIONS[$((choice - 1))]}"; break
            fi
            echo "  Invalid choice."
        done
    fi
fi

case "$KV_CACHE_PRESET" in
    q8_0)  CACHE_K="q8_0"; CACHE_V="q8_0"; BYTES_PER_TOKEN=30 ;;
    mixed) CACHE_K="q8_0"; CACHE_V="q4_0"; BYTES_PER_TOKEN=22 ;;
    q4_0)  CACHE_K="q4_0"; CACHE_V="q4_0"; BYTES_PER_TOKEN=15 ;;
    *) echo "ERROR: Unknown KV cache preset '$KV_CACHE_PRESET'. Use: q8_0, mixed, q4_0"; exit 1 ;;
esac

# ===================================================================
# Resolve context target
# ===================================================================
CTX_TARGET_OPTIONS=("262k" "512k" "768k" "1m")
CTX_TARGET_LABELS=(
    "262K   native context, no quality loss"
    "512K   YaRN scaling — modest quality loss"
    "768K   YaRN scaling — noticeable quality loss"
    "1M     YaRN scaling — significant quality loss at context edges"
)

if [ -z "$CONTEXT_TARGET" ]; then
    if [ "$AUTO_YES" = "true" ]; then
        CONTEXT_TARGET="262k"
    else
        echo ""
        echo "Select context target:"
        echo ""
        for i in "${!CTX_TARGET_OPTIONS[@]}"; do
            default_marker=""
            [ "$i" = "0" ] && default_marker=" (default)"
            echo "  $((i+1))) ${CTX_TARGET_LABELS[$i]}${default_marker}"
        done
        echo ""
        while true; do
            read -p "Context [1-${#CTX_TARGET_OPTIONS[@]}] (Enter for default): " choice
            if [ -z "$choice" ]; then CONTEXT_TARGET="262k"; break; fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#CTX_TARGET_OPTIONS[@]} ]; then
                CONTEXT_TARGET="${CTX_TARGET_OPTIONS[$((choice - 1))]}"; break
            fi
            echo "  Invalid choice."
        done
    fi
fi

case "$CONTEXT_TARGET" in
    262k) MAX_CONTEXT=262144;  USE_YARN=false ;;
    512k) MAX_CONTEXT=524288;  USE_YARN=true ;;
    768k) MAX_CONTEXT=786432;  USE_YARN=true ;;
    1m)   MAX_CONTEXT=1048576; USE_YARN=true ;;
    *) echo "ERROR: Unknown context target '$CONTEXT_TARGET'. Use: 262k, 512k, 768k, 1m"; exit 1 ;;
esac

GGUF_FILENAME=$(get_gguf_filename "$FILE_PATTERN" "$QUANT")
VRAM_EST=$(get_vram_estimate $MODEL_IDX "$QUANT")

# ===================================================================
# Confirm
# ===================================================================
echo ""
echo "════════════════════════════════════════════"
echo "  llama.cpp Setup"
echo "════════════════════════════════════════════"
echo ""
YARN_DISPLAY="off"
[ "$USE_YARN" = "true" ] && YARN_DISPLAY="ENABLED (target: $CONTEXT_TARGET)"

echo "  Model:       $MODEL_NAME"
echo "  Quant:       $QUANT ($VRAM_EST)"
echo "  File:        $GGUF_FILENAME"
echo "  HF repo:     $HF_REPO"
echo "  Context:     $CONTEXT_LENGTH tokens (max: $MAX_CONTEXT)"
echo "  KV cache:    $KV_CACHE_PRESET (K=$CACHE_K, V=$CACHE_V — ~${BYTES_PER_TOKEN} bytes/token)"
echo "  YaRN:        $YARN_DISPLAY"
echo "  Parallel:    $PARALLEL slots"
echo "  Port:        $PORT"
echo "  Thinking:    $([ "$ENABLE_THINKING" = "true" ] && echo "ENABLED" || echo "disabled")"
echo "  Directory:   $WORK_DIR"
echo ""

if [ "$AUTO_YES" = "false" ]; then
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# ===================================================================
# [1/7] System packages
# ===================================================================
echo ""
echo "[1/7] Installing system packages..."

sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip curl openssl cmake build-essential git > /dev/null 2>&1
pip install --quiet huggingface-hub[cli] 2>/dev/null || pip install --quiet --break-system-packages huggingface-hub[cli] 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"

echo "  Done."

# ===================================================================
# [2/7] Check NVIDIA GPUs
# ===================================================================
echo "[2/7] Checking NVIDIA GPUs..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
TOTAL_VRAM=$((GPU_MEM * GPU_COUNT))

echo "  Found ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM} MiB each, ${TOTAL_VRAM} MiB total)"

# Build tensor split string for multi-GPU
TENSOR_SPLIT=""
if [ "$GPU_COUNT" -gt 1 ]; then
    SPLIT_VAL=$(python3 -c "print(','.join(['1'] * $GPU_COUNT))")
    TENSOR_SPLIT="--tensor-split $SPLIT_VAL"
    echo "  Multi-GPU: tensor split $SPLIT_VAL"
fi

# Auto-size context window based on available VRAM
if [ "$CONTEXT_LENGTH" = "auto" ]; then
    MODEL_GB=$(echo "$VRAM_EST" | grep -oE '[0-9]+' | head -1)
    MODEL_MB=$((MODEL_GB * 1024))
    AVAILABLE_MB=$((TOTAL_VRAM - MODEL_MB - 2048))

    if [ "$AVAILABLE_MB" -le 0 ]; then
        echo "ERROR: Model ($VRAM_EST) is too large for ${TOTAL_VRAM} MiB VRAM."
        exit 1
    fi

    # BYTES_PER_TOKEN set by KV cache preset: q8_0=30, mixed=22, q4_0=15
    MAX_CTX=$(( (AVAILABLE_MB * 1024 * 1024) / BYTES_PER_TOKEN ))

    # Clamp to context target max and round down to nearest 4096
    if [ "$MAX_CTX" -gt "$MAX_CONTEXT" ]; then MAX_CTX=$MAX_CONTEXT; fi
    MAX_CTX=$(( (MAX_CTX / 4096) * 4096 ))

    if [ "$MAX_CTX" -lt 8192 ]; then
        echo "ERROR: Only ${MAX_CTX} tokens of context fit. Model too large for this GPU."
        exit 1
    fi
    if [ "$MAX_CTX" -lt 32768 ]; then
        echo "  WARNING: Only ${MAX_CTX} tokens of context available. Consider a smaller quant."
    fi

    CONTEXT_LENGTH=$MAX_CTX
    echo "  Context auto-sized: ${CONTEXT_LENGTH} tokens (${AVAILABLE_MB} MiB for KV cache, ${BYTES_PER_TOKEN} bytes/token)"
fi

# ===================================================================
# [3/7] Install llama.cpp
# ===================================================================
echo "[3/7] Building llama.cpp with CUDA support..."

mkdir -p "$WORK_DIR"

LLAMA_SRC="$WORK_DIR/llama.cpp"

if [ -d "$LLAMA_SRC" ]; then
    echo "  Source already cloned."
else
    echo "  Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_SRC"
fi

LATEST_TAG=$(git -C "$LLAMA_SRC" tag --sort=-v:refname 2>/dev/null | head -1)
if [ -z "$LATEST_TAG" ]; then
    LATEST_TAG=$(git -C "$LLAMA_SRC" rev-parse --short HEAD)
fi

if [ -f "$WORK_DIR/bin/llama-server" ] && [ -f "$WORK_DIR/bin/.version" ] && [ "$(cat "$WORK_DIR/bin/.version")" = "$LATEST_TAG" ]; then
    echo "  llama.cpp $LATEST_TAG already built."
else
    echo "  Building $LATEST_TAG (this takes a few minutes)..."

    # Use the ML image's CUDA toolkit (not the apt package which is often outdated)
    CUDA_COMPILER=""
    if [ -f /usr/local/cuda/bin/nvcc ]; then
        CUDA_COMPILER="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
        export PATH="/usr/local/cuda/bin:$PATH"
    elif command -v nvcc &>/dev/null; then
        CUDA_COMPILER="-DCMAKE_CUDA_COMPILER=$(which nvcc)"
    else
        echo "ERROR: nvcc not found. Install CUDA toolkit or use a GCP ML image."
        exit 1
    fi

    echo "  Configuring cmake..."
    cmake -B "$LLAMA_SRC/build" -S "$LLAMA_SRC" \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        $CUDA_COMPILER \
        > /dev/null 2>&1
    echo "  Compiling ($(nproc) threads)..."
    cmake --build "$LLAMA_SRC/build" --config Release -j "$(nproc)" 2>&1 | \
        awk 'NR % 20 == 0 { printf "  [%d files compiled]\n", NR; fflush() } END { printf "  [%d files total]\n", NR }'

    mkdir -p "$WORK_DIR/bin"
    cp "$LLAMA_SRC/build/bin/llama-server" "$WORK_DIR/bin/"
    echo "$LATEST_TAG" > "$WORK_DIR/bin/.version"
    echo "  Built and installed to $WORK_DIR/bin/"
fi

if [ ! -f "$WORK_DIR/bin/llama-server" ]; then
    echo "ERROR: llama-server binary not found after build."
    echo "Check build logs above for errors."
    exit 1
fi

# ===================================================================
# [4/7] Download model
# ===================================================================
echo "[4/7] Downloading model..."
echo "  Repo: $HF_REPO"
echo "  File: $GGUF_FILENAME"

mkdir -p "$WORK_DIR/models"

MODEL_PATH="$WORK_DIR/models/$GGUF_FILENAME"

if [ -f "$MODEL_PATH" ]; then
    echo "  Model already downloaded."
else
    HF_CMD="hf"
    if ! command -v hf &>/dev/null; then
        HF_CMD="huggingface-cli"
    fi
    $HF_CMD download "$HF_REPO" "$GGUF_FILENAME" \
        --local-dir "$WORK_DIR/models" || {
        echo ""
        echo "  Download failed. Try manually:"
        echo "    hf download $HF_REPO $GGUF_FILENAME --local-dir $WORK_DIR/models"
        exit 1
    }
    echo "  Model downloaded."
fi

# Download vision adapter if available
MMPROJ_PATH=""
if [ -n "$MMPROJ_FILE" ]; then
    MMPROJ_PATH="$WORK_DIR/models/$MMPROJ_FILE"
    if [ -f "$MMPROJ_PATH" ]; then
        echo "  Vision adapter already downloaded."
    else
        echo "  Downloading vision adapter ($MMPROJ_FILE)..."
        $HF_CMD download "$HF_REPO" "$MMPROJ_FILE" \
            --local-dir "$WORK_DIR/models" || {
            echo "  Vision adapter download failed (non-fatal, vision disabled)."
            MMPROJ_PATH=""
        }
    fi
fi

# ===================================================================
# [5/7] Generate API key
# ===================================================================
echo "[5/7] Generating API key..."

API_KEY_FILE="$WORK_DIR/.api_key"
if [ -f "$API_KEY_FILE" ]; then
    echo "  API key already exists."
else
    openssl rand -hex 32 > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
    echo "  API key generated."
fi

API_KEY=$(cat "$API_KEY_FILE")

# ===================================================================
# [6/7] Create systemd service
# ===================================================================
echo "[6/7] Creating systemd service..."

# Create system prompt file
cat > "$WORK_DIR/system_prompt.txt" << 'PROMPTEOF'
You are a pragmatic, context-aware engineering assistant. Adapt your approach based on the task type. Prioritize correctness, maintainability, and explicit reasoning over verbosity.

For new projects: ask clarifying questions, propose minimal architecture before coding.
For existing code: analyze before suggesting changes, preserve existing patterns, flag breaking changes.
For quick tasks: be concise, provide code with minimal context.

Never hallucinate APIs or framework behavior. Flag uncertain logic. Prefer clear code over clever optimizations.
PROMPTEOF

THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "false" ]; then
    THINKING_FLAG="--chat-template-kwargs '{\"enable_thinking\":false}'"
fi

MMPROJ_FLAG=""
if [ -n "$MMPROJ_PATH" ] && [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_FLAG="--mmproj $MMPROJ_PATH"
fi

YARN_FLAG=""
if [ "$USE_YARN" = "true" ]; then
    YARN_FLAG="--rope-scaling yarn"
fi

SERVICE_NAME="llamacpp"

sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=llama.cpp Server - $MODEL_NAME ($QUANT)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$WORK_DIR/bin/llama-server \\
    --model $MODEL_PATH \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --ctx-size $CONTEXT_LENGTH \\
    --parallel $PARALLEL \\
    --gpu-layers all \\
    --cache-type-k $CACHE_K \\
    --cache-type-v $CACHE_V \\
    $YARN_FLAG \\
    $TENSOR_SPLIT \\
    --api-key $API_KEY \\
    --alias $MODEL_ALIAS \\
    $MMPROJ_FLAG \\
    --jinja \\
    $THINKING_FLAG \\
    --metrics
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME.service

echo "  Service created: $SERVICE_NAME.service"

# ===================================================================
# [7/7] Create test script
# ===================================================================
echo "[7/7] Creating test script..."

cat > "$WORK_DIR/test_api.sh" << 'TESTEOF'
#!/bin/bash

PORT=PORT_PLACEHOLDER
API_KEY=$(cat API_KEY_FILE_PLACEHOLDER)
BASE_URL="http://localhost:$PORT"

echo "Testing llama.cpp Server..."
echo "========================================="
echo ""

PASS=0
FAIL=0

# 1. Health check
echo "1. Health check..."
HEALTH=$(curl -s --max-time 10 "$BASE_URL/health")
if echo "$HEALTH" | grep -qi "ok\|alive\|status"; then
    echo "  PASS"
    PASS=$((PASS + 1))
else
    echo "  FAIL: $HEALTH"
    FAIL=$((FAIL + 1))
fi

# 2. Chat completion
echo ""
echo "2. Chat completion..."
RESPONSE=$(curl -s --max-time 120 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"Write a Python function to check if a number is prime. Be concise."}],"max_tokens":200}')
CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null)
if [ -n "$CONTENT" ]; then
    echo "  Response: $CONTENT..."
    echo "  PASS"
    PASS=$((PASS + 1))
else
    echo "  FAIL: No valid response"
    echo "  $RESPONSE" | head -5
    FAIL=$((FAIL + 1))
fi

# 3. Auth enforcement
echo ""
echo "3. Auth enforcement..."
NOAUTH=$(curl -s --max-time 10 \
    -H "Content-Type: application/json" \
    "$BASE_URL/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":10}')
if echo "$NOAUTH" | grep -qi "unauthorized\|error\|401"; then
    echo "  PASS"
    PASS=$((PASS + 1))
else
    echo "  FAIL: Unauthenticated request was not rejected"
    FAIL=$((FAIL + 1))
fi

# 4. Streaming
echo ""
echo "4. Streaming..."
STREAM=$(curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    "$BASE_URL/v1/chat/completions" \
    -d '{"messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":20,"stream":true}')
if echo "$STREAM" | grep -q "data:"; then
    echo "  PASS"
    PASS=$((PASS + 1))
else
    echo "  FAIL: No streaming data received"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "========================================="
if [ $FAIL -eq 0 ]; then
    echo "All $PASS tests passed."
else
    echo "$PASS passed, $FAIL failed."
    exit 1
fi
TESTEOF

sed -i "s|PORT_PLACEHOLDER|$PORT|g" "$WORK_DIR/test_api.sh"
sed -i "s|API_KEY_FILE_PLACEHOLDER|$API_KEY_FILE|g" "$WORK_DIR/test_api.sh"
chmod +x "$WORK_DIR/test_api.sh"

echo "  Test script created at $WORK_DIR/test_api.sh"

# ===================================================================
# Completion
# ===================================================================
echo ""
echo "════════════════════════════════════════════"
echo "  Installation Complete!"
echo "════════════════════════════════════════════"
echo ""
echo "  Model:     $MODEL_NAME ($QUANT)"
echo "  Backend:   llama.cpp $LATEST_TAG"
echo "  Service:   $SERVICE_NAME"
echo "  Directory: $WORK_DIR"
echo "  API key:   $API_KEY_FILE"
echo "  Thinking:  $([ "$ENABLE_THINKING" = "true" ] && echo "ENABLED" || echo "disabled")"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Start the server:"
echo "     sudo systemctl start $SERVICE_NAME.service"
echo ""
echo "  2. Watch startup logs:"
echo "     sudo journalctl -u $SERVICE_NAME.service -f"
echo ""
echo "  3. Test the API:"
echo "     cd $WORK_DIR && ./test_api.sh"
echo ""
echo "  4. Get your VM's external IP:"
echo "     curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
echo ""
echo "  5. Get your API key:"
echo "     cat $API_KEY_FILE"
echo ""
echo "  Configure GCP firewall to allow port $PORT:"
echo "     gcloud compute firewall-rules create allow-llamacpp \\"
echo "       --allow=tcp:$PORT \\"
echo "       --source-ranges=YOUR_IP/32"
echo ""
echo "  Restart after VM reboot:"
echo "     ./setup_llamacpp.sh --start-only"
echo ""
