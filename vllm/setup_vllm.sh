#!/bin/bash

# Unified vLLM setup script for Qwen 27B on Google Cloud GPU VMs.
# Supports native (systemd) and Docker deployment modes.
#
# Usage:
#   ./setup_vllm.sh                              # interactive native setup
#   ./setup_vllm.sh --docker --quant FP8 --yes   # automated Docker setup
#   ./setup_vllm.sh --start-only                 # restart systemd service
#   ./setup_vllm.sh --docker --start-only        # restart Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common library
if [ -f "$SCRIPT_DIR/vllm_common.sh" ]; then
    source "$SCRIPT_DIR/vllm_common.sh"
else
    echo "ERROR: vllm_common.sh not found in $SCRIPT_DIR"
    exit 1
fi

# Docker constants
VLLM_IMAGE="vllm/vllm-openai"
VLLM_VERSION="v0.21.0"

# ===================================================================
# Mode-specific functions: Docker
# ===================================================================

do_start_only_docker() {
    if [ ! -f "$HOME/vllm-docker/.env" ]; then
        print_error "$HOME/vllm-docker/.env not found. Run full setup first."
        exit 1
    fi
    local work_dir="$HOME/vllm-docker"
    print_info "Starting Docker container..."
    cd "$work_dir" && sudo docker compose up -d
    print_success "Container started"

    local port
    port=$(grep "^PORT=" "$work_dir/.env" 2>/dev/null | cut -d= -f2)
    port=${port:-8000}
    wait_for_healthy "$port"

    local api_key
    api_key=$(cat "$work_dir/.api_key" 2>/dev/null || echo "")
    local served_name
    served_name=$(grep "^SERVED_NAME=" "$work_dir/.env" 2>/dev/null | cut -d= -f2)
    if [ -n "$served_name" ]; then
        warmup_vllm "$port" "$served_name" "$api_key"
    fi

    echo ""
    echo "Check logs: cd $work_dir && docker compose logs -f"
}

install_docker_prereqs() {
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
}

pull_vllm_image() {
    echo "[2/4] Pulling vLLM image ($VLLM_IMAGE:$VLLM_VERSION)..."
    sudo docker pull "$VLLM_IMAGE:$VLLM_VERSION" || {
        print_error "Failed to pull $VLLM_IMAGE:$VLLM_VERSION"
        exit 1
    }
    echo "  Done."
}

configure_docker_env() {
    echo "[3/4] Configuring..."
    mkdir -p "$WORK_DIR"

    API_KEY=$(generate_api_key "$WORK_DIR")

    cat > "$WORK_DIR/.env" << EOF
API_KEY=$API_KEY
MODEL=$HF_MODEL
SERVED_NAME=$SERVED_NAME
PORT=$PORT
TENSOR_PARALLEL=$TENSOR_PARALLEL
MAX_MODEL_LEN=$MAX_MODEL_LEN
GPU_MEM_UTIL=$GPU_MEM_UTIL
VLLM_VERSION=$VLLM_VERSION
HF_CACHE=$HOME/.cache/huggingface
EXTRA_FLAGS=$EXTRA_FLAGS
EOF

    cp "$SCRIPT_DIR/docker-compose.yml" "$WORK_DIR/docker-compose.yml"
    echo "  Configured."
}

start_docker_container() {
    echo "[4/4] Starting container..."
    cd "$WORK_DIR"
    sudo docker compose down 2>/dev/null || true
    sudo docker compose up -d
}

print_docker_completion() {
    local external_ip
    external_ip=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "(unknown)")

    echo ""
    print_header "Deployment Complete!"
    echo ""
    echo "  Model:    $MODEL_DISPLAY"
    echo "  MTP:      $([ "$ENABLE_MTP" = "true" ] && echo "ENABLED" || echo "disabled")"
    echo "  API:      http://$external_ip:$PORT/v1/"
    echo "  API Key:  $API_KEY"
    echo "  Model ID: $SERVED_NAME"
    echo ""
    echo "  Logs:     cd $WORK_DIR && docker compose logs -f"
    echo "  Stop:     cd $WORK_DIR && sudo docker compose down"
    echo "  Restart:  ./setup_vllm.sh --docker --start-only"
    echo ""
}

# ===================================================================
# Mode-specific functions: Native (systemd)
# ===================================================================

do_start_only_native() {
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        print_error "Service $SERVICE_NAME not found. Run full setup first."
        exit 1
    fi
    print_info "Starting $SERVICE_NAME.service..."
    sudo systemctl start "$SERVICE_NAME.service"
    print_success "Service started"

    wait_for_healthy "$PORT"

    local api_key=""
    [ -f "$WORK_DIR/.api_key" ] && api_key=$(cat "$WORK_DIR/.api_key")
    warmup_vllm "$PORT" "$SERVED_NAME" "$api_key"

    echo ""
    echo "Check status:  sudo systemctl status $SERVICE_NAME.service"
    echo "View logs:     sudo journalctl -u $SERVICE_NAME.service -f"
}

install_native() {
    update_system
    install_essential_packages
    verify_nvidia_drivers
    check_gpu_memory "$MIN_VRAM" "$MIN_GPUS"
    create_venv "$WORK_DIR"

    source "$WORK_DIR/venv/bin/activate"
    install_uv
    install_pytorch
    install_vllm "true"
    install_hf_dependencies
    huggingface_login
    download_model "$HF_MODEL" "$MODEL_SIZE"
}

configure_native() {
    API_KEY=$(generate_api_key "$WORK_DIR")

    create_vllm_start_script \
        "$WORK_DIR" \
        "$HF_MODEL" \
        "$TENSOR_PARALLEL" \
        "$MAX_MODEL_LEN" \
        "$GPU_MEM_UTIL" \
        "$SERVED_NAME" \
        "$EXTRA_FLAGS" \
        "$ENABLE_TOOL_CALLING" \
        "" \
        "$WORK_DIR/.api_key" \
        "$PORT"

    create_systemd_service \
        "$SERVICE_NAME" \
        "$WORK_DIR" \
        "$MODEL_DISPLAY vLLM Server"

    configure_firewall "$PORT"

    create_test_script \
        "$WORK_DIR" \
        "$SERVED_NAME" \
        "$MODEL_DISPLAY"
}

print_native_completion() {
    echo ""
    print_header "Installation Complete!"
    echo ""
    echo "  Model:     $MODEL_DISPLAY"
    echo "  MTP:       $([ "$ENABLE_MTP" = "true" ] && echo "ENABLED" || echo "disabled")"
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
}

# ===================================================================
# Main orchestration
# ===================================================================

init_quant_registry
parse_vllm_args "$@"

# Handle --start-only early exit
if [ "$START_ONLY" = "true" ]; then
    if [ "$DEPLOY_MODE" = "docker" ]; then
        do_start_only_docker
    else
        set_derived_config
        do_start_only_native
    fi
    exit 0
fi

# Shared config resolution
resolve_quant_selection
set_derived_config
prompt_mtp
assemble_extra_flags

if [ "$DEPLOY_MODE" = "docker" ]; then
    confirm_installation "Docker"
    install_docker_prereqs
    pull_vllm_image
    configure_docker_env
    start_docker_container
    wait_for_healthy "$PORT"
    warmup_vllm "$PORT" "$SERVED_NAME" "$API_KEY"
    print_docker_completion
else
    confirm_installation "Native"
    install_native
    configure_native
    print_native_completion
fi
