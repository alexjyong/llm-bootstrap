#!/bin/bash

# vLLM setup script for Qwen 27B on Google Cloud GPU VMs.
# Supports multiple quantization levels with an interactive picker or CLI flags.
#
# Sets up:
#   - vLLM inference server
#   - Model download from Hugging Face
#   - Built-in API key auth (no nginx needed)
#   - systemd service
#
# Usage:
#   ./setup_vllm.sh                          # interactive quant picker
#   ./setup_vllm.sh --quant FP8              # skip TUI
#   ./setup_vllm.sh --quant AWQ --yes        # fully non-interactive
#   ./setup_vllm.sh --start-only             # skip install, just start services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common library
if [ -f "$SCRIPT_DIR/vllm_common.sh" ]; then
    source "$SCRIPT_DIR/vllm_common.sh"
else
    echo "ERROR: vllm_common.sh not found in $SCRIPT_DIR"
    exit 1
fi

# ===================================================================
# Quantization registry
# ===================================================================
QUANT_NAMES=(
    "AWQ (INT4)"
    "NVFP4 (NVIDIA FP4)"
    "FP8"
    "BF16 (full precision)"
)
QUANT_KEYS=("AWQ" "NVFP4" "FP8" "BF16")
QUANT_DESCS=(
    "Smallest integer quant. Fits on 1x L4 (24GB). Near Q5_K_M quality."
    "NVIDIA 4-bit float. Fits on 1x L4 (24GB). ~99% of BF16 quality. Best for multi-user."
    "Best quality-per-dollar. Fits on 2x L4 (48GB). Nearly lossless."
    "Full precision. Requires 1x A100 80GB. Baseline quality."
)
QUANT_VRAM=("~17 GB" "~14 GB" "~27 GB" "~54 GB")
QUANT_GPU_CONFIGS=(
    "1x L4 (24GB) — g2-standard-12"
    "1x L4 (24GB) — g2-standard-12"
    "2x L4 (48GB) — g2-standard-24"
    "1x A100 80GB — a2-highgpu-1g"
)
DEFAULT_QUANT_IDX=2  # FP8

# Model configs per quantization
QUANT_HF_MODELS=(
    "cyankiwi/Qwen3.6-27B-AWQ-INT4"
    "unsloth/Qwen3.6-27B-NVFP4"
    "Qwen/Qwen3.6-27B-FP8"
    "Qwen/Qwen3.6-27B"
)
QUANT_VLLM_FLAGS=(
    "--quantization awq"
    "--kv-cache-dtype fp8_e5m2"
    "--kv-cache-dtype fp8_e5m2"
    ""
)
QUANT_TENSOR_PARALLEL=(1 1 2 1)
QUANT_MAX_MODEL_LEN=(32768 32768 65536 65536)
QUANT_GPU_MEM_UTIL=(0.92 0.92 0.95 0.95)
QUANT_MIN_VRAM_MB=(24576 24576 49152 81920)
QUANT_MIN_GPU_COUNT=(1 1 2 1)
QUANT_MODEL_SIZES=("~17GB" "~14GB" "~27GB" "~54GB")

# ===================================================================
# Parse arguments
# ===================================================================
AUTO_YES=false
START_ONLY=false
QUANT_ARG=""
ENABLE_TOOL_CALLING=false
PORT=8000

show_usage() {
    cat << 'EOF'
vLLM Setup for Qwen 27B

Usage: ./setup_vllm.sh [options]

Options:
  --quant <AWQ|NVFP4|FP8|BF16>  Quantization level (skip interactive picker)
  --yes, -y                      Skip all prompts (non-interactive)
  --start-only                   Skip installation, just start existing service
  --enable-tool-calling          Enable function/tool calling support
  --port <port>                  API port (default: 8000)

Quantization options:
  AWQ    INT4, ~17GB — fits on 1x L4 (24GB)
  NVFP4  NVIDIA FP4, ~14GB — fits on 1x L4 (24GB), best for multi-user
  FP8    8-bit float, ~27GB — fits on 2x L4 (48GB) (default)
  BF16   Full precision, ~54GB — needs 1x A100 80GB

Examples:
  ./setup_vllm.sh                              # interactive picker
  ./setup_vllm.sh --quant FP8 --yes            # automated FP8 setup
  ./setup_vllm.sh --quant NVFP4 --yes          # smallest, multi-user optimized
  ./setup_vllm.sh --start-only                 # restart after VM reboot

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
        --quant)
            QUANT_ARG="$2"
            shift 2
            ;;
        --enable-tool-calling|--tool-calling)
            ENABLE_TOOL_CALLING=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
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
    SERVICE_NAME="vllm-qwen-27b"
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        print_error "Service $SERVICE_NAME not found. Run full setup first."
        exit 1
    fi
    print_info "Starting $SERVICE_NAME.service..."
    sudo systemctl start "$SERVICE_NAME.service"
    print_success "Service started"
    echo ""
    echo "Check status:  sudo systemctl status $SERVICE_NAME.service"
    echo "View logs:     sudo journalctl -u $SERVICE_NAME.service -f"
    exit 0
fi

# ===================================================================
# Resolve quantization selection
# ===================================================================
resolve_quant() {
    local arg="$1"
    local upper=$(echo "$arg" | tr '[:lower:]' '[:upper:]')

    for i in "${!QUANT_KEYS[@]}"; do
        if [ "${QUANT_KEYS[$i]}" = "$upper" ]; then
            QUANT_IDX=$i
            return 0
        fi
    done

    echo "ERROR: Unknown quantization '$arg'"
    echo "Available: ${QUANT_KEYS[*]}"
    exit 1
}

if [ -n "$QUANT_ARG" ]; then
    resolve_quant "$QUANT_ARG"
elif [ "$AUTO_YES" = "true" ]; then
    QUANT_IDX=$DEFAULT_QUANT_IDX
else
    # Interactive quant picker
    echo ""
    print_header "vLLM Setup — Qwen 27B"
    echo ""
    echo "Select quantization level:"
    echo ""
    for i in "${!QUANT_NAMES[@]}"; do
        default_marker=""
        if [ "$i" = "$DEFAULT_QUANT_IDX" ]; then
            default_marker=" (default)"
        fi
        echo "  $((i+1))) ${QUANT_NAMES[$i]}  ${QUANT_VRAM[$i]}${default_marker}"
        echo "     ${QUANT_DESCS[$i]}"
        echo "     GPU: ${QUANT_GPU_CONFIGS[$i]}"
        echo ""
    done
    while true; do
        read -p "Quantization [1-${#QUANT_NAMES[@]}] (Enter for default): " choice
        if [ -z "$choice" ]; then
            QUANT_IDX=$DEFAULT_QUANT_IDX
            break
        fi
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#QUANT_NAMES[@]} ]; then
            QUANT_IDX=$((choice - 1))
            break
        fi
        echo "  Invalid choice. Enter 1-${#QUANT_NAMES[@]} or press Enter for default."
    done
fi

# ===================================================================
# Set derived configuration
# ===================================================================
QUANT_KEY="${QUANT_KEYS[$QUANT_IDX]}"
HF_MODEL="${QUANT_HF_MODELS[$QUANT_IDX]}"
VLLM_QUANT_FLAGS="${QUANT_VLLM_FLAGS[$QUANT_IDX]}"
TENSOR_PARALLEL="${QUANT_TENSOR_PARALLEL[$QUANT_IDX]}"
MAX_MODEL_LEN="${QUANT_MAX_MODEL_LEN[$QUANT_IDX]}"
GPU_MEMORY_UTIL="${QUANT_GPU_MEM_UTIL[$QUANT_IDX]}"
MIN_VRAM="${QUANT_MIN_VRAM_MB[$QUANT_IDX]}"
MIN_GPUS="${QUANT_MIN_GPU_COUNT[$QUANT_IDX]}"
MODEL_SIZE="${QUANT_MODEL_SIZES[$QUANT_IDX]}"

SERVED_NAME="qwen3-27b"
WORK_DIR="$HOME/qwen-27b-vllm"
SERVICE_NAME="vllm-qwen-27b"
MODEL_DISPLAY="Qwen3.6-27B ($QUANT_KEY)"

# ===================================================================
# Confirm and install
# ===================================================================
echo ""
print_header "$MODEL_DISPLAY vLLM Setup"
echo ""
echo "  Model:          $HF_MODEL"
echo "  Quantization:   ${QUANT_NAMES[$QUANT_IDX]}"
echo "  Model size:     ${QUANT_VRAM[$QUANT_IDX]}"
echo "  GPUs:           ${QUANT_GPU_CONFIGS[$QUANT_IDX]}"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo "  Max context:    $MAX_MODEL_LEN tokens"
if [ "$ENABLE_TOOL_CALLING" = "true" ]; then
    echo "  Tool calling:   ENABLED"
fi
echo ""

if [ "$AUTO_YES" = "false" ]; then
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
else
    echo "Auto-confirmed (--yes flag)"
fi

# ===================================================================
# Installation
# ===================================================================
update_system
install_essential_packages
verify_nvidia_drivers
check_gpu_memory $MIN_VRAM $MIN_GPUS
create_venv "$WORK_DIR"

# Install dependencies in venv
source "$WORK_DIR/venv/bin/activate"
install_uv
install_pytorch
install_vllm "true"  # always use nightly for latest Qwen support
install_hf_dependencies
huggingface_login
download_model "$HF_MODEL" "$MODEL_SIZE"

# Generate API key
API_KEY=$(generate_api_key "$WORK_DIR")

# Create server start script
create_vllm_start_script \
    "$WORK_DIR" \
    "$HF_MODEL" \
    "$TENSOR_PARALLEL" \
    "$MAX_MODEL_LEN" \
    "$GPU_MEMORY_UTIL" \
    "$SERVED_NAME" \
    "$VLLM_QUANT_FLAGS" \
    "$ENABLE_TOOL_CALLING" \
    "" \
    "$WORK_DIR/.api_key" \
    "$PORT"

# Create systemd service
create_systemd_service \
    "$SERVICE_NAME" \
    "$WORK_DIR" \
    "$MODEL_DISPLAY vLLM Server"

configure_firewall "$PORT"

# Create test script
create_test_script \
    "$WORK_DIR" \
    "$SERVED_NAME" \
    "$MODEL_DISPLAY"

# ===================================================================
# Completion
# ===================================================================
echo ""
print_header "Installation Complete!"
echo ""
echo "  Model:     $MODEL_DISPLAY"
echo "  Service:   $SERVICE_NAME"
echo "  Directory: $WORK_DIR"
echo "  API key:   $WORK_DIR/.api_key"
echo ""
print_info "Next steps:"
echo ""
echo "  1. Start the server:"
echo "     sudo systemctl start $SERVICE_NAME.service"
echo ""
echo "  2. Wait for startup (watch for 'Uvicorn running on http://0.0.0.0:$PORT'):"
echo "     sudo journalctl -u $SERVICE_NAME.service -f"
echo ""
echo "  3. Test the API:"
echo "     cd $WORK_DIR && python3 test_api.py"
echo ""
echo "  4. Get your VM's external IP:"
echo "     curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
echo ""
echo "  5. Get your API key:"
echo "     cat $WORK_DIR/.api_key"
echo ""
print_warning "Configure GCP firewall to allow port $PORT:"
echo "     gcloud compute firewall-rules create allow-vllm \\"
echo "       --allow=tcp:$PORT \\"
echo "       --source-ranges=YOUR_IP/32"
echo ""
echo "  Restart after VM reboot:"
echo "     ./setup_vllm.sh --start-only"
echo ""
print_success "Setup complete!"
