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
)
MODEL_HF_REPOS=(
    "unsloth/Qwen3.6-27B-GGUF"
    "unsloth/Qwen3.6-35B-A3B-GGUF"
    "unsloth/gemma-4-31b-it-GGUF"
)
MODEL_FILE_PATTERNS=(
    "Qwen3.6-27B"
    "Qwen3.6-35B-A3B"
    "gemma-4-31B-it"
)
MODEL_MMPROJ_FILES=(
    "mmproj-Qwen3.6-27B-BF16.gguf"
    "mmproj-Qwen3.6-35B-A3B-BF16.gguf"
    ""
)
MODEL_ALIASES=("qwen3.6-27b" "qwen3.6-35b-a3b" "gemma4-31b")
MODEL_DEFAULT_QUANTS=("Q6_K" "Q4_K_M" "Q6_K")

QUANT_OPTIONS=("Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0")

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
IDENTIFIER=""

show_usage() {
    cat << 'EOF'
Docker llama.cpp Setup

Usage: ./setup_docker.sh [options]

Options:
  --model <number>            Model (1=27B, 2=35B-A3B)
  --quant <Q3_K_M|...|Q8_0>  Quantization
  --kv-cache <preset>         KV cache preset: q8_0, mixed, q4_0 (default: q8_0)
  --context-target <target>   Context target: 262k, 512k, 768k, 1m (default: 262k)
                              Targets above 262k enable YaRN rope scaling
  --yes, -y                   Skip prompts
  --start-only                Restart existing container
  --port <port>               API port (default: 8080)
  --context-length <N>        Exact context window in tokens (overrides --context-target)
  --parallel <N>              Concurrent slots (default: 3)
  --mtp                       Enable Multi-Token Prediction (27B only, ~2x faster generation)
  --identifier <name>         Custom model ID for API requests (default: model name)

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
        --identifier) IDENTIFIER="$2"; shift 2 ;;
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
    exit 0
fi

# ===================================================================
# Resolve model
# ===================================================================
if [ -n "$MODEL_ARG" ]; then
    MODEL_IDX=-1
    for i in "${!MODEL_NAMES[@]}"; do
        if [ "$MODEL_ARG" = "$((i+1))" ] || [[ "${MODEL_NAMES[$i],,}" == *"${MODEL_ARG,,}"* ]]; then
            MODEL_IDX=$i
            break
        fi
    done
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

# Apply context target as cap (--context-length overrides if explicitly set)
if [ "$CONTEXT_LENGTH" = "262144" ]; then
    CONTEXT_LENGTH=$MAX_CONTEXT
fi

EXTRA_FLAGS=""
if [ "$USE_YARN" = "true" ]; then
    EXTRA_FLAGS="--rope-scaling yarn"
fi
if [ "$ENABLE_MTP" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --spec-type draft-mtp --spec-draft-n-max 3"
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

API_KEY_FILE="$WORK_DIR/.api_key"
if [ ! -f "$API_KEY_FILE" ]; then
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
echo "  API:      http://$EXTERNAL_IP:$PORT/v1/"
echo "  API Key:  $API_KEY"
echo "  Model ID: $MODEL_ALIAS"
echo ""
echo "  Logs:     cd $WORK_DIR && docker compose logs -f"
echo "  Stop:     cd $WORK_DIR && sudo docker compose down"
echo "  Restart:  cd $WORK_DIR && sudo docker compose up -d"
echo "  Test:     cd $WORK_DIR && ./test_api.sh"
echo ""
