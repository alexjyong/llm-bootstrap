#!/bin/bash

# Docker-based llama.cpp setup for GCP GPU VMs.
# Pulls a pre-built llama-server image — no compilation needed.
#
# Usage:
#   ./setup_docker.sh                              # interactive picker
#   ./setup_docker.sh --model 1 --quant Q6_K --yes # non-interactive
#   ./setup_docker.sh --start-only                 # restart container

set -e

# ===================================================================
# Model registry
# ===================================================================
MODEL_NAMES=(
    "Qwen 3.6-27B (dense)"
    "Qwen 3.6-35B-A3B (MoE)"
    "Gemma 4 31B (dense)"
    "Muse Glimmer 30B (dense, vision)"
)
MODEL_HF_REPOS=(
    "unsloth/Qwen3.6-27B-GGUF"
    "unsloth/Qwen3.6-35B-A3B-GGUF"
    "unsloth/gemma-4-31b-it-GGUF"
    "unsloth/Muse-Glimmer-30B-GGUF"
)
MODEL_FILE_PATTERNS=(
    "Qwen3.6-27B"
    "Qwen3.6-35B-A3B"
    "gemma-4-31B-it"
    "Muse-Glimmer-30B"
)
MODEL_MMPROJ_FILES=(
    "mmproj-BF16.gguf"
    "mmproj-BF16.gguf"
    "mmproj-BF16.gguf"
    "mmproj-Muse-Glimmer-30B-BF16.gguf"
)
MODEL_ALIASES=("qwen3.6-27b" "qwen3.6-35b-a3b" "gemma4-31b" "muse-glimmer-30b")
MODEL_DEFAULT_QUANTS=("Q6_K" "Q4_K_M" "Q6_K" "UD-Q4_K_XL")

# Muse Glimmer ships Unsloth Dynamic quants (UD-*_XL) instead of the
# classic K-quants, so it gets its own quant list (applied after model
# resolution below).
QUANT_OPTIONS=("Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0")
MUSE_QUANT_OPTIONS=("UD-Q2_K_XL" "UD-Q3_K_XL" "UD-Q4_K_XL" "UD-Q6_K_XL" "UD-Q8_K_XL" "Q8_0")

DOCKER_IMAGE="ghcr.io/alexjyong/llm-bootstrap/llama-server:latest"
WORK_DIR="$HOME/llama-docker"

# ===================================================================
# Parse arguments
# ===================================================================
AUTO_YES=false
START_ONLY=false
MODEL_ARG=""
QUANT=""
PORT=8080
CONTEXT_LENGTH=262144
PARALLEL=3
KV_CACHE_PRESET=""
CONTEXT_TARGET=""
ENABLE_MTP=false
ENABLE_DFLASH=false
IDENTIFIER=""
ENABLE_NGROK=false
API_KEY_ARG=""
ENABLE_FIXED_TEMPLATE=false

# DFlash draft model source (self-converted from the primary source, not a
# third-party GGUF — see docs/dflash-research.md). Tune here, not via a flag.
DFLASH_SPEC_DRAFT_N_MAX=12

show_usage() {
    cat << 'EOF'
Docker llama.cpp Setup

Usage: ./setup_docker.sh [options]

Options:
  --model <number>            Model (1=27B, 2=35B-A3B, 3=Gemma31B, 4=MuseGlimmer30B)
  --quant <Q3_K_M|...|Q8_0>  Quantization
  --kv-cache <preset>         KV cache preset: q8_0, mixed, q4_0 (default: q8_0)
  --context-target <target>   Context target: 262k, 512k, 768k, 1m (default: 262k)
                              Muse Glimmer (4) instead offers: 131k (native), 262k (YaRN)
                              Targets above native context enable YaRN rope scaling
  --yes, -y                   Skip prompts
  --start-only                Restart existing container
  --port <port>               API port (default: 8080)
  --context-length <N>        Exact context window in tokens (overrides --context-target)
  --parallel <N>              Concurrent slots (default: 3)
  --mtp                       Enable Multi-Token Prediction (27B only, ~2x faster generation)
  --dflash                    Enable DFlash speculative decoding (27B only, ~3.75x faster generation)
                              Self-converts a draft model from the primary source on first run
                              (one-time, adds a few minutes). Mutually exclusive with --mtp.
  --identifier <name>         Custom model ID for API requests (default: model name)
  --ngrok                     Expose the server via an ngrok tunnel (needs NGROK_AUTHTOKEN)
  --api-key <key>             Use this API key instead of generating a random one
                              (also reads LLM_API_KEY from the environment)
  --fixed-chat-template       Use froggeric's community chat template fix (Qwen models only)
                              https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --yes|-y) AUTO_YES=true; shift ;;
        --start-only) START_ONLY=true; shift ;;
        --model) MODEL_ARG="$2"; shift 2 ;;
        --quant) QUANT="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --kv-cache) KV_CACHE_PRESET="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --context-target) CONTEXT_TARGET="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --mtp) ENABLE_MTP=true; shift ;;
        --dflash) ENABLE_DFLASH=true; shift ;;
        --identifier) IDENTIFIER="$2"; shift 2 ;;
        --ngrok) ENABLE_NGROK=true; shift ;;
        --api-key) API_KEY_ARG="$2"; shift 2 ;;
        --fixed-chat-template) ENABLE_FIXED_TEMPLATE=true; shift ;;
        --help|-h) show_usage; exit 0 ;;
        *) echo "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# ===================================================================
# Handle --start-only
# ===================================================================
if [ "$START_ONLY" = "true" ]; then
    if [ ! -f "$WORK_DIR/.env" ]; then
        echo "ERROR: $WORK_DIR/.env not found. Run full setup first."
        exit 1
    fi
    echo "Starting container..."
    cd "$WORK_DIR" && sudo docker compose up -d
    echo "Started. Check logs: docker compose logs -f"

    if [ "$ENABLE_NGROK" = "true" ]; then
        ENV_PORT=$(grep "^PORT=" "$WORK_DIR/.env" 2>/dev/null | cut -d= -f2)
        source "$HOME/ngrok.sh"
        ngrok_setup_if_enabled "true" "${ENV_PORT:-$PORT}" "$AUTO_YES" || true
    fi

    exit 0
fi

# ===================================================================
# Resolve model
# ===================================================================
if [ -n "$MODEL_ARG" ]; then
    MODEL_IDX=-1
    # Exact numeric match first, in its own pass — otherwise e.g. "--model 2"
    # can wrongly fuzzy-match model 1's name ("Qwen 3.6-27B" contains a "2")
    # before the loop ever reaches the real index-2 candidate.
    for i in "${!MODEL_NAMES[@]}"; do
        if [ "$MODEL_ARG" = "$((i+1))" ]; then
            MODEL_IDX=$i
            break
        fi
    done
    # A purely numeric arg that didn't exact-match is out of range — never
    # fuzzy-match digits ("5" would substring-match "Qwen 3.6-35B-A3B" and
    # silently deploy the wrong model).
    if [ "$MODEL_IDX" = "-1" ] && [[ "$MODEL_ARG" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Model number $MODEL_ARG out of range (1-${#MODEL_NAMES[@]}). Available:"
        for i in "${!MODEL_NAMES[@]}"; do echo "  $((i+1))) ${MODEL_NAMES[$i]}"; done
        exit 1
    fi
    if [ "$MODEL_IDX" = "-1" ]; then
        for i in "${!MODEL_NAMES[@]}"; do
            if [[ "${MODEL_NAMES[$i],,}" == *"${MODEL_ARG,,}"* ]]; then
                MODEL_IDX=$i
                break
            fi
        done
    fi
    if [ "$MODEL_IDX" = "-1" ]; then
        echo "ERROR: Unknown model '$MODEL_ARG'. Available:"
        for i in "${!MODEL_NAMES[@]}"; do echo "  $((i+1))) ${MODEL_NAMES[$i]}"; done
        exit 1
    fi
elif [ "$AUTO_YES" = "true" ]; then
    MODEL_IDX=0
else
    echo ""
    echo "Select model:"
    for i in "${!MODEL_NAMES[@]}"; do
        echo "  $((i+1))) ${MODEL_NAMES[$i]}"
    done
    echo ""
    while true; do
        read -p "Model [1-${#MODEL_NAMES[@]}] (Enter for default): " choice
        if [ -z "$choice" ]; then MODEL_IDX=0; break; fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#MODEL_NAMES[@]} ]; then
            MODEL_IDX=$((choice - 1)); break
        fi
        echo "  Invalid choice."
    done
fi

HF_REPO="${MODEL_HF_REPOS[$MODEL_IDX]}"
FILE_PATTERN="${MODEL_FILE_PATTERNS[$MODEL_IDX]}"
MMPROJ_FILE="${MODEL_MMPROJ_FILES[$MODEL_IDX]}"
MODEL_ALIAS="${MODEL_ALIASES[$MODEL_IDX]}"
[ -n "$IDENTIFIER" ] && MODEL_ALIAS="$IDENTIFIER"

if [ "$MODEL_IDX" = "3" ]; then
    QUANT_OPTIONS=("${MUSE_QUANT_OPTIONS[@]}")
fi

if [ "$ENABLE_FIXED_TEMPLATE" = "true" ] && [ "$MODEL_IDX" -gt 1 ]; then
    echo "ERROR: --fixed-chat-template only supports Qwen models (1, 2), not Gemma (3) or Muse Glimmer (4)."
    exit 1
fi

# ===================================================================
# MTP prompt (only for Qwen 3.6-27B)
# ===================================================================
if [ "$MODEL_IDX" = "0" ] && [ "$ENABLE_MTP" = "false" ] && [ "$AUTO_YES" = "false" ]; then
    echo ""
    echo "Enable Multi-Token Prediction (MTP)?"
    echo "  ~2x faster generation using built-in draft prediction heads"
    echo ""
    echo "  1) No   (standard model weights)"
    echo "  2) Yes  (use MTP model weights)"
    echo ""
    while true; do
        read -p "MTP [1-2] (Enter for default): " choice
        if [ -z "$choice" ] || [ "$choice" = "1" ]; then break; fi
        if [ "$choice" = "2" ]; then ENABLE_MTP=true; break; fi
        echo "  Invalid choice."
    done
fi

if [ "$ENABLE_MTP" = "true" ] && [ "$MODEL_IDX" != "0" ]; then
    echo "ERROR: MTP is only supported for Qwen 3.6-27B (model 1)."
    exit 1
fi

if [ "$ENABLE_MTP" = "true" ]; then
    HF_REPO="unsloth/Qwen3.6-27B-MTP-GGUF"
    FILE_PATTERN="Qwen3.6-27B"
    MMPROJ_FILE=""
    QUANT_OPTIONS=("Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0" "BF16")
    MODEL_DEFAULT_QUANTS[0]="Q6_K"
fi

# ===================================================================
# DFlash restrictions (only for Qwen 3.6-27B, mutually exclusive with MTP)
# ===================================================================
if [ "$ENABLE_DFLASH" = "true" ] && [ "$MODEL_IDX" != "0" ]; then
    echo "ERROR: DFlash is only supported for Qwen 3.6-27B (model 1)."
    exit 1
fi

if [ "$ENABLE_DFLASH" = "true" ] && [ "$ENABLE_MTP" = "true" ]; then
    echo "ERROR: --mtp and --dflash are mutually exclusive (both are speculative-decoding modes)."
    exit 1
fi

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
        for i in "${!QUANT_OPTIONS[@]}"; do
            default_marker=""
            [ "${QUANT_OPTIONS[$i]}" = "$DEFAULT_QUANT" ] && default_marker=" (default)"
            echo "  $((i+1))) ${QUANT_OPTIONS[$i]}${default_marker}"
        done
        echo ""
        while true; do
            read -p "Quant [1-${#QUANT_OPTIONS[@]}] (Enter for default): " choice
            if [ -z "$choice" ]; then QUANT="$DEFAULT_QUANT"; break; fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#QUANT_OPTIONS[@]} ]; then
                QUANT="${QUANT_OPTIONS[$((choice - 1))]}"; break
            fi
            echo "  Invalid choice."
        done
    fi
fi

MODEL_FILE="${FILE_PATTERN}-${QUANT}.gguf"

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

# DFlash needs f16 KV cache — quantized KV cache measured a 7x slowdown in
# draft verification speed in third-party testing. This has no VRAM-aware
# auto-sizing safety net in this script (context is a flat default/flag,
# same as without DFlash) — forced f16 costs roughly double the VRAM per
# token of the q8_0 default, so pass --context-length explicitly if it OOMs.
if [ "$ENABLE_DFLASH" = "true" ]; then
    echo "  NOTE: --dflash forces f16 KV cache (quantized KV cache causes a measured 7x"
    echo "        slowdown in draft verification), costing ~2x the VRAM per token of"
    echo "        $KV_CACHE_PRESET. Lower --context-length explicitly if the server OOMs."
    CACHE_K="f16"; CACHE_V="f16"; BYTES_PER_TOKEN=60
fi

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
DEFAULT_CTX_TARGET="262k"
NATIVE_CTX=262144

# Muse Glimmer: native context is 131072, documented ceiling is 262144
# (via RoPE scaling) — https://unsloth.ai/docs/models/muse-glimmer
if [ "$MODEL_IDX" = "3" ]; then
    CTX_TARGET_OPTIONS=("131k" "262k")
    CTX_TARGET_LABELS=(
        "131K   native context, no quality loss"
        "262K   YaRN scaling (2x) — documented ceiling, modest quality loss"
    )
    DEFAULT_CTX_TARGET="131k"
    NATIVE_CTX=131072
fi

if [ -z "$CONTEXT_TARGET" ]; then
    if [ "$AUTO_YES" = "true" ]; then
        CONTEXT_TARGET="$DEFAULT_CTX_TARGET"
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
            if [ -z "$choice" ]; then CONTEXT_TARGET="$DEFAULT_CTX_TARGET"; break; fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#CTX_TARGET_OPTIONS[@]} ]; then
                CONTEXT_TARGET="${CTX_TARGET_OPTIONS[$((choice - 1))]}"; break
            fi
            echo "  Invalid choice."
        done
    fi
fi

if [ "$MODEL_IDX" = "3" ]; then
    case "$CONTEXT_TARGET" in
        131k) MAX_CONTEXT=131072; USE_YARN=false ;;
        262k) MAX_CONTEXT=262144; USE_YARN=true ;;
        *) echo "ERROR: Unknown context target '$CONTEXT_TARGET' for Muse Glimmer. Use: 131k, 262k"; exit 1 ;;
    esac
else
    case "$CONTEXT_TARGET" in
        262k) MAX_CONTEXT=262144;  USE_YARN=false ;;
        512k) MAX_CONTEXT=524288;  USE_YARN=true ;;
        768k) MAX_CONTEXT=786432;  USE_YARN=true ;;
        1m)   MAX_CONTEXT=1048576; USE_YARN=true ;;
        *) echo "ERROR: Unknown context target '$CONTEXT_TARGET'. Use: 262k, 512k, 768k, 1m"; exit 1 ;;
    esac
fi

# Apply context target as cap (--context-length overrides if explicitly set)
if [ "$CONTEXT_LENGTH" = "262144" ]; then
    CONTEXT_LENGTH=$MAX_CONTEXT
else
    # An explicit --context-length above native context implies YaRN, otherwise
    # llama.cpp just clamps --ctx-size back down to the trained length.
    if [ "$CONTEXT_LENGTH" -gt "$NATIVE_CTX" ]; then
        USE_YARN=true
        MAX_CONTEXT=$CONTEXT_LENGTH
    fi
fi

EXTRA_FLAGS=""
if [ "$USE_YARN" = "true" ]; then
    # Scale factor = target / native trained context (Qwen: 262144, Muse: 131072)
    ROPE_SCALE=$(python3 -c "print($MAX_CONTEXT / $NATIVE_CTX)")
    EXTRA_FLAGS="--rope-scaling yarn --rope-scale $ROPE_SCALE --yarn-orig-ctx $NATIVE_CTX"
    # Muse Glimmer: llama.cpp clamps --ctx-size to the trained length read
    # from GGUF metadata — override it (same trick verified to 1M context:
    # https://www.reddit.com/r/LocalLLaMA — Muse's global layers are NoPE, so
    # YaRN stretching degrades far less than on full-RoPE architectures)
    EXTRA_FLAGS="$EXTRA_FLAGS --override-kv muse-glimmer.context_length=int:$MAX_CONTEXT"
fi
if [ "$ENABLE_MTP" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --spec-type draft-mtp --spec-draft-n-max 3"
fi
DFLASH_MODEL_FILE="qwen3.6-27b-dflash-bf16.gguf"
if [ "$ENABLE_DFLASH" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS -md /models/$DFLASH_MODEL_FILE --spec-type draft-dflash --spec-draft-n-max $DFLASH_SPEC_DRAFT_N_MAX"
fi
if [ "$ENABLE_FIXED_TEMPLATE" = "true" ]; then
    mkdir -p "$WORK_DIR/models"
    source "$HOME/chat_templates.sh"
    if download_fixed_chat_template "$WORK_DIR/models/chat_template.jinja"; then
        EXTRA_FLAGS="$EXTRA_FLAGS --chat-template-file /models/chat_template.jinja"
    fi
fi
# Meta's recommended Muse Glimmer generation settings
# (https://unsloth.ai/docs/models/muse-glimmer)
if [ "$MODEL_IDX" = "3" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --temp 1.0 --top-p 0.95 --top-k 64"
fi

YARN_DISPLAY="off"
[ "$USE_YARN" = "true" ] && YARN_DISPLAY="ENABLED (target: $CONTEXT_TARGET)"

echo ""
echo "════════════════════════════════════════════"
echo "  Docker llama.cpp Setup"
echo "════════════════════════════════════════════"
echo ""
echo "  Model:     ${MODEL_NAMES[$MODEL_IDX]}"
echo "  Quant:     $QUANT"
echo "  File:      $MODEL_FILE"
echo "  Context:   $CONTEXT_LENGTH tokens"
echo "  KV cache:  $KV_CACHE_PRESET (K=$CACHE_K, V=$CACHE_V — ~${BYTES_PER_TOKEN} bytes/token)"
echo "  YaRN:      $YARN_DISPLAY"
echo "  Parallel:  $PARALLEL slots"
echo "  Port:      $PORT"
echo "  Image:     $DOCKER_IMAGE"
echo "  MTP:       $([ "$ENABLE_MTP" = "true" ] && echo "ENABLED (spec-draft-n-max: 3)" || echo "disabled")"
echo "  DFlash:    $([ "$ENABLE_DFLASH" = "true" ] && echo "ENABLED (spec-draft-n-max: $DFLASH_SPEC_DRAFT_N_MAX, self-converted draft)" || echo "disabled")"
echo "  Chat tmpl: $([ "$ENABLE_FIXED_TEMPLATE" = "true" ] && echo "fixed (froggeric)" || echo "default")"
[ "$MODEL_IDX" = "3" ] && echo "  Sampling:  temp=1.0 top-p=0.95 top-k=64 (Meta recommended)"
echo ""

if [ "$AUTO_YES" = "false" ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && echo "Cancelled." && exit 0
fi

# ===================================================================
# [1/4] Check Docker + NVIDIA
# ===================================================================
echo ""
echo "[1/4] Checking Docker and GPU..."

if ! command -v docker &>/dev/null; then
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed."
fi

if ! sudo docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    if ! command -v nvidia-container-cli &>/dev/null; then
        echo "  Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null 2>&1
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  Docker + GPU ready."

# ===================================================================
# [2/4] Download model
# ===================================================================
echo "[2/4] Downloading model..."

mkdir -p "$WORK_DIR/models"

sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip > /dev/null
pip3 install --quiet huggingface-hub[cli] 2>/dev/null || pip3 install --quiet --break-system-packages huggingface-hub[cli] || {
    echo "ERROR: Failed to install huggingface-hub. Try manually:"
    echo "  pip3 install huggingface-hub[cli]"
    exit 1
}
export PATH="$HOME/.local/bin:$PATH"

HF_CMD="hf"
if ! command -v hf &>/dev/null; then
    HF_CMD="huggingface-cli"
    if ! command -v huggingface-cli &>/dev/null; then
        echo "ERROR: Could not find hf CLI after install. Try manually:"
        echo "  pip3 install huggingface-hub[cli]"
        exit 1
    fi
fi
echo "  Using $HF_CMD"

NEED_DOWNLOAD=true
if [ -f "$WORK_DIR/models/$MODEL_FILE" ]; then
    if [ -f "$WORK_DIR/models/.hf_repo" ] && [ "$(cat "$WORK_DIR/models/.hf_repo")" = "$HF_REPO" ]; then
        echo "  Model already downloaded."
        NEED_DOWNLOAD=false
    else
        echo "  Existing model is from a different repo — re-downloading..."
        rm -f "$WORK_DIR/models/$MODEL_FILE"
    fi
fi
if [ "$NEED_DOWNLOAD" = "true" ]; then
    echo "  Downloading $MODEL_FILE from $HF_REPO..."
    $HF_CMD download "$HF_REPO" --local-dir "$WORK_DIR/models" -- "$MODEL_FILE" || {
        echo "  Download failed."
        exit 1
    }
    echo "$HF_REPO" > "$WORK_DIR/models/.hf_repo"
fi

if [ -n "$MMPROJ_FILE" ] && [ ! -f "$WORK_DIR/models/$MMPROJ_FILE" ]; then
    echo "  Downloading vision adapter..."
    $HF_CMD download "$HF_REPO" "$MMPROJ_FILE" --local-dir "$WORK_DIR/models" || true
fi

if [ "$ENABLE_DFLASH" = "true" ]; then
    echo "  Setting up DFlash draft model (self-converted from the primary source)..."
    DFLASH_CONVERT_SRC="$WORK_DIR/.dflash_convert_src"
    if [ -d "$DFLASH_CONVERT_SRC" ]; then
        echo "    llama.cpp conversion source already cloned."
    else
        echo "    Cloning llama.cpp (conversion scripts only, not built — the server itself"
        echo "    runs from the prebuilt Docker image)..."
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$DFLASH_CONVERT_SRC"
    fi
    source "$HOME/dflash_convert.sh"
    convert_dflash_draft "$DFLASH_CONVERT_SRC" "$WORK_DIR/models/$DFLASH_MODEL_FILE" "$WORK_DIR/models/.dflash_marker" || {
        echo "ERROR: DFlash draft conversion failed. Aborting."
        exit 1
    }
fi

echo "  Done."

# ===================================================================
# [3/4] Pull image + configure
# ===================================================================
echo "[3/4] Pulling Docker image..."

# Authenticate with GitHub Container Registry if GH_TOKEN is available
GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -n "$GH_TOKEN" ]; then
    echo "$GH_TOKEN" | sudo docker login ghcr.io -u USERNAME --password-stdin 2>/dev/null
    echo "  Authenticated with ghcr.io"
elif ! sudo docker pull "$DOCKER_IMAGE" 2>/dev/null; then
    echo "  Private registry requires authentication."
    echo "  Set GH_TOKEN env var with a GitHub token (read:packages scope)."
    echo "  Or building locally instead..."
fi

sudo docker pull "$DOCKER_IMAGE" 2>/dev/null || {
    echo "  Pull failed. Building locally (this takes ~15 min)..."
    sudo docker build -t "$DOCKER_IMAGE" "$(dirname "$0")"
}

# The pulled image can succeed but still predate llama.cpp's DFlash merge —
# it's only rebuilt via a manual workflow_dispatch, not automatically, so it
# can silently go stale. Check for real instead of assuming, and self-heal
# by building locally from current master if it's missing.
if [ "$ENABLE_DFLASH" = "true" ]; then
    if ! sudo docker run --rm --gpus all "$DOCKER_IMAGE" --help 2>&1 | grep -q "draft-dflash"; then
        echo "  WARNING: Pulled image predates llama.cpp's DFlash support (it's only"
        echo "  rebuilt on-demand, not automatically). Rebuilding locally from current"
        echo "  master (this takes ~15 min)..."
        sudo docker build -t "$DOCKER_IMAGE" "$(dirname "$0")"
        if ! sudo docker run --rm --gpus all "$DOCKER_IMAGE" --help 2>&1 | grep -q "draft-dflash"; then
            echo "ERROR: Rebuilt image still doesn't support --spec-type draft-dflash. Aborting."
            exit 1
        fi
    fi
fi

# Same staleness guard for Muse Glimmer: the pulled image can predate
# llama.cpp's Muse Glimmer architecture support. The arch string is
# "muse-glimmer" and lives in the shared libs (libllama.so), not the
# llama-server binary — so grep both.
if [ "$MODEL_IDX" = "3" ]; then
    if ! sudo docker run --rm --entrypoint grep "$DOCKER_IMAGE" -qra "muse-glimmer" /usr/local/lib /usr/local/bin; then
        echo "  WARNING: Pulled image predates llama.cpp's Muse Glimmer support (it's"
        echo "  only rebuilt on-demand, not automatically). Rebuilding locally from"
        echo "  current master (this takes ~15 min)..."
        sudo docker build -t "$DOCKER_IMAGE" "$(dirname "$0")"
        if ! sudo docker run --rm --entrypoint grep "$DOCKER_IMAGE" -qra "muse-glimmer" /usr/local/lib /usr/local/bin; then
            echo "ERROR: Rebuilt image still doesn't support the muse-glimmer architecture. Aborting."
            exit 1
        fi
    fi
fi

API_KEY_FILE="$WORK_DIR/.api_key"
CUSTOM_API_KEY="${API_KEY_ARG:-${LLM_API_KEY:-}}"
if [ -n "$CUSTOM_API_KEY" ]; then
    echo "$CUSTOM_API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
elif [ ! -f "$API_KEY_FILE" ]; then
    openssl rand -hex 32 > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
fi
API_KEY=$(cat "$API_KEY_FILE")

cat > "$WORK_DIR/.env" << EOF
DOCKER_IMAGE=$DOCKER_IMAGE
API_KEY=$API_KEY
MODELS_DIR=$WORK_DIR/models
MODEL_FILE=$MODEL_FILE
MMPROJ_FLAG=$([ -n "$MMPROJ_FILE" ] && echo "--mmproj /models/$MMPROJ_FILE" || echo "")
MODEL_ALIAS=$MODEL_ALIAS
PORT=$PORT
CONTEXT_LENGTH=$CONTEXT_LENGTH
PARALLEL=$PARALLEL
CACHE_TYPE_K=$CACHE_K
CACHE_TYPE_V=$CACHE_V
EXTRA_FLAGS=$EXTRA_FLAGS
EOF

cp "$(dirname "$0")/docker-compose.yml" "$WORK_DIR/docker-compose.yml"

echo "  Configured."

# ===================================================================
# [4/4] Start container
# ===================================================================
echo "[4/4] Starting container..."

cd "$WORK_DIR"
sudo docker compose down 2>/dev/null || true
sudo docker compose up -d

echo ""
echo "  Waiting for server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$PORT/health | grep -q "ok"; then
        echo "  Server ready!"
        break
    fi
    [ "$i" = "30" ] && echo "  WARNING: Server still starting. Check: docker compose logs -f"
    sleep 5
done

EXTERNAL_IP=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "(unknown)")

if [ "$ENABLE_NGROK" = "true" ]; then
    echo ""
    source "$HOME/ngrok.sh"
    ngrok_setup_if_enabled "true" "$PORT" "$AUTO_YES" || true
fi

# ===================================================================
# Generate test script
# ===================================================================
cat > "$WORK_DIR/test_api.sh" << 'TESTEOF'
#!/bin/bash

PORT=PORT_PLACEHOLDER
API_KEY=$(cat API_KEY_FILE_PLACEHOLDER)
BASE_URL="http://localhost:$PORT"

echo "Testing llama.cpp Docker Server..."
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

echo ""
echo "════════════════════════════════════════════"
echo "  Deployment Complete!"
echo "════════════════════════════════════════════"
echo ""
echo "  Model:    ${MODEL_NAMES[$MODEL_IDX]} ($QUANT)"
echo "  MTP:      $([ "$ENABLE_MTP" = "true" ] && echo "ENABLED" || echo "disabled")"
echo "  DFlash:   $([ "$ENABLE_DFLASH" = "true" ] && echo "ENABLED" || echo "disabled")"
echo "  Chat tmpl: $([ "$ENABLE_FIXED_TEMPLATE" = "true" ] && echo "fixed (froggeric)" || echo "default")"
echo "  API:      http://$EXTERNAL_IP:$PORT/v1/"
if [ -f "$HOME/.ngrok_url" ]; then
    echo "  ngrok:    $(cat "$HOME/.ngrok_url")/v1/"
fi
echo "  API Key:  $API_KEY"
echo "  Model ID: $MODEL_ALIAS"
echo ""
echo "  Logs:     cd $WORK_DIR && docker compose logs -f"
echo "  Stop:     cd $WORK_DIR && sudo docker compose down"
echo "  Restart:  cd $WORK_DIR && sudo docker compose up -d"
echo "  Test:     cd $WORK_DIR && ./test_api.sh"
echo ""
