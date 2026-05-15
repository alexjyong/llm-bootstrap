#!/bin/bash

# LLM deployment and VM management tool.
# Combines deploying backends to VMs and managing running services.
#
# Usage:
#   ./llm.sh deploy                                    # interactive wizard
#   ./llm.sh deploy VM_NAME --backend llamacpp --yes   # non-interactive
#   ./llm.sh list                                      # list all VMs
#   ./llm.sh creds VM_NAME                             # show API credentials
#   ./llm.sh config VM_NAME context-length 262144      # change settings
#   ./llm.sh stop VM_NAME                              # stop VM

set -euo pipefail

PROJECT="${GCP_PROJECT:-your-project-id}"

# ===================================================================
# Help
# ===================================================================
show_usage() {
    cat << 'EOF'
LLM deployment and VM management tool.

Usage: ./llm.sh <command> [options]

Deploy:
  deploy [vm-name] [options]      Deploy a backend to a VM (interactive wizard if no args)
    --backend <name>              Backend: llamacpp, llamacpp-docker, vllm
    --yes, -y                     Skip all prompts
    All other flags (--model, --quant, --port, --context-length, --start-only, etc.)
    are passed through to the setup script.

Manage:
  list                            List all VMs (name, zone, status, IP)
  creds  <vm-name>                Show IP, port, API key, model ID, and base URL
  logs   <vm-name>                Show server logs (last 50 lines)
  config <vm-name> <key> <value>  Update a server setting and restart
  stop   <vm-name>                Stop a VM (keeps disk, stops billing)
  start  <vm-name>                Start a stopped VM
  resume <vm-name>                Start VM + restart LLM service (auto-detects backend)
    --backend <name>              Skip auto-detection, use this backend
  test   <vm-name>                Test the API (health, models, chat completion)
  ssh    <vm-name>                SSH into a VM
  ip     <vm-name>                Print a VM's external IP
  delete <vm-name>                Delete a VM and its disk (irreversible)

Config keys:
  context-length <N>              Context window in tokens (e.g. 262144)
  parallel <N>                    Concurrent request slots (e.g. 3)
  port <N>                        API port (e.g. 8080)
  thinking <on|off>               Enable/disable thinking mode

All commands auto-detect the VM's zone. Use --zone to override.

Examples:
  ./llm.sh deploy                                              # interactive wizard
  ./llm.sh deploy my-llm --backend llamacpp --quant Q6_K --yes # non-interactive
  ./llm.sh list
  ./llm.sh creds my-llm
  ./llm.sh config my-llm context-length 262144
  ./llm.sh stop my-llm
  ./llm.sh start my-llm
  ./llm.sh resume my-llm                                      # resume (auto-detects backend)
  ./llm.sh resume my-llm --backend llamacpp-docker             # resume with explicit backend
EOF
}

# ===================================================================
# Shared helpers
# ===================================================================
find_zone() {
    gcloud compute instances list \
        --project="$PROJECT" \
        --filter="name=$1" \
        --format="value(zone)" 2>/dev/null | head -1
}

resolve_zone() {
    local name="$1"
    local zone="$2"
    if [ -n "$zone" ]; then
        echo "$zone"
        return
    fi
    zone=$(find_zone "$name")
    if [ -z "$zone" ]; then
        echo "ERROR: VM '$name' not found in project $PROJECT" >&2
        exit 1
    fi
    echo "$zone"
}

parse_vm_args() {
    VM_NAME=""
    VM_ZONE=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --zone) VM_ZONE="$2"; shift 2 ;;
            -*) echo "Unknown option: $1"; exit 1 ;;
            *) VM_NAME="$1"; shift ;;
        esac
    done
    if [ -z "$VM_NAME" ]; then
        echo "ERROR: VM name required"
        show_usage
        exit 1
    fi
}

get_vm_status() {
    gcloud compute instances describe "$1" \
        --project="$PROJECT" \
        --zone="$2" \
        --format="value(status)" 2>/dev/null
}

get_vm_ip() {
    gcloud compute instances describe "$1" \
        --project="$PROJECT" \
        --zone="$2" \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null
}

wait_for_ssh() {
    local name="$1" zone="$2"
    echo "Waiting for SSH..."
    for i in $(seq 1 30); do
        if gcloud compute ssh "$name" \
            --zone="$zone" \
            --project="$PROJECT" \
            --command="echo ready" 2>/dev/null | grep -q "ready"; then
            echo "  SSH ready."
            return 0
        fi
        echo "  Attempt $i/30 — retrying in 10s..."
        sleep 10
    done
    echo "ERROR: SSH not available after 5 minutes."
    exit 1
}

detect_backend() {
    local name="$1" zone="$2"
    gcloud compute ssh "$name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --command="
            if [ -f /etc/systemd/system/llamacpp.service ]; then echo llamacpp
            elif [ -f ~/llama-docker/.env ]; then echo llamacpp-docker
            elif ls /etc/systemd/system/vllm-*.service >/dev/null 2>&1; then echo vllm
            fi
        " 2>/dev/null
}

scp_to_vm() {
    local name="$1" zone="$2"
    shift 2
    gcloud compute scp "$@" "${name}:~" \
        --zone="$zone" \
        --project="$PROJECT"
}

ssh_command() {
    local name="$1" zone="$2" cmd="$3"
    gcloud compute ssh "$name" \
        --zone="$zone" \
        --project="$PROJECT" \
        --ssh-flag="-o ServerAliveInterval=30" \
        --ssh-flag="-o ServerAliveCountMax=10" \
        --command="$cmd"
}

# ===================================================================
# Deploy: TUI wizard
# ===================================================================
BACKEND_NAMES=("llamacpp" "llamacpp-docker" "vllm")
BACKEND_LABELS=(
    "llama.cpp — direct inference, no daemon, native thinking control"
    "llama.cpp (Docker) — pre-built, no compile step, fastest deploy"
    "vLLM — concurrent users, high throughput"
)
MODEL_NAMES=(
    "Qwen 3.6-27B (dense)"
    "Qwen 3.6-35B-A3B (MoE)"
    "Qwen 3.5-122B-A10B (MoE)"
)
MODEL_ARGS=("1" "2" "3")
QUANT_OPTIONS_LLAMACPP=("Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K" "Q8_0")
QUANT_OPTIONS_VLLM=("AWQ" "FP8" "BF16")

pick_vm() {
    echo ""
    echo "Fetching VMs..."
    local vm_list
    vm_list=$(gcloud compute instances list \
        --project="$PROJECT" \
        --format="table[no-heading](name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP,machineType.machine_type().basename())" \
        2>/dev/null)

    if [ -z "$vm_list" ]; then
        echo "ERROR: No VMs found in project $PROJECT"
        echo "Create one first: ./create_gpu_vm.sh"
        exit 1
    fi

    echo ""
    echo "Select VM:"
    echo ""
    local i=1
    local names=() zones=()
    while IFS= read -r line; do
        local name zone status ip machine
        read -r name zone status ip machine <<< "$line"
        names+=("$name")
        zones+=("$zone")
        printf "  %d) %-16s %-18s %-10s %-16s %s\n" "$i" "$name" "$zone" "$status" "$ip" "$machine"
        i=$((i + 1))
    done <<< "$vm_list"
    echo ""

    while true; do
        read -p "VM [1-${#names[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#names[@]} ]; then
            VM_NAME="${names[$((choice - 1))]}"
            VM_ZONE="${zones[$((choice - 1))]}"
            break
        fi
        echo "  Invalid choice."
    done
}

pick_backend() {
    echo ""
    echo "Select backend:"
    echo ""
    for i in "${!BACKEND_NAMES[@]}"; do
        echo "  $((i + 1))) ${BACKEND_LABELS[$i]}"
    done
    echo ""
    while true; do
        read -p "Backend [1-${#BACKEND_NAMES[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#BACKEND_NAMES[@]} ]; then
            BACKEND="${BACKEND_NAMES[$((choice - 1))]}"
            break
        fi
        echo "  Invalid choice."
    done
}

pick_model() {
    if [ "$BACKEND" = "vllm" ]; then return; fi
    local has_model=false
    for flag in "${SETUP_FLAGS[@]}"; do
        [ "$flag" = "--model" ] && has_model=true
    done
    if [ "$has_model" = "true" ]; then return; fi

    echo ""
    echo "Select model:"
    echo ""
    for i in "${!MODEL_NAMES[@]}"; do
        echo "  $((i + 1))) ${MODEL_NAMES[$i]}"
    done
    echo ""
    while true; do
        read -p "Model [1-${#MODEL_NAMES[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#MODEL_NAMES[@]} ]; then
            SETUP_FLAGS+=("--model" "${MODEL_ARGS[$((choice - 1))]}")
            break
        fi
        echo "  Invalid choice."
    done
}

pick_quant() {
    local has_quant=false
    for flag in "${SETUP_FLAGS[@]}"; do
        [ "$flag" = "--quant" ] && has_quant=true
    done
    if [ "$has_quant" = "true" ]; then return; fi

    local quants=()
    case "$BACKEND" in
        vllm) quants=("${QUANT_OPTIONS_VLLM[@]}") ;;
        *) quants=("${QUANT_OPTIONS_LLAMACPP[@]}") ;;
    esac

    echo ""
    echo "Select quantization:"
    echo ""
    for i in "${!quants[@]}"; do
        echo "  $((i + 1))) ${quants[$i]}"
    done
    echo ""
    while true; do
        read -p "Quant [1-${#quants[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#quants[@]} ]; then
            SETUP_FLAGS+=("--quant" "${quants[$((choice - 1))]}")
            break
        fi
        echo "  Invalid choice."
    done
}

# ===================================================================
# Deploy command
# ===================================================================
do_deploy() {
    VM_NAME=""
    BACKEND=""
    VM_ZONE=""
    SETUP_FLAGS=()
    AUTO_YES=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend) BACKEND="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
            --zone) VM_ZONE="$2"; shift 2 ;;
            --yes|-y) AUTO_YES=true; SETUP_FLAGS+=("--yes"); shift ;;
            --model|--quant|--port|--context-length|--identifier)
                SETUP_FLAGS+=("$1" "$2"); shift 2 ;;
            --enable-tool-calling|--tool-calling|--start-only)
                SETUP_FLAGS+=("$1"); shift ;;
            -*) SETUP_FLAGS+=("$1"); shift ;;
            *)
                if [ -z "$VM_NAME" ]; then VM_NAME="$1"
                else SETUP_FLAGS+=("$1"); fi
                shift ;;
        esac
    done

    # TUI wizard if needed
    if [ -z "$VM_NAME" ]; then pick_vm; fi
    if [ -z "$BACKEND" ]; then
        pick_backend
        pick_model
        pick_quant
    fi

    # Validate backend
    case "$BACKEND" in
        llamacpp|llamacpp-docker|vllm) ;;
        *)
            echo "ERROR: Unknown backend '$BACKEND'"
            echo "Available: llamacpp, llamacpp-docker, vllm"
            exit 1 ;;
    esac

    # Resolve zone
    if [ -z "$VM_ZONE" ]; then
        echo "Looking up zone for $VM_NAME..."
        VM_ZONE=$(find_zone "$VM_NAME")
        if [ -z "$VM_ZONE" ]; then
            echo "ERROR: VM '$VM_NAME' not found in project $PROJECT"
            exit 1
        fi
        echo "  Found in $VM_ZONE"
    fi

    # Check VM status
    STATUS=$(get_vm_status "$VM_NAME" "$VM_ZONE")
    if [ "$STATUS" != "RUNNING" ]; then
        echo "VM $VM_NAME is $STATUS."
        if [ "$AUTO_YES" = "true" ]; then
            echo "Starting VM..."
            gcloud compute instances start "$VM_NAME" --zone="$VM_ZONE" --project="$PROJECT"
        else
            read -p "Start it? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                gcloud compute instances start "$VM_NAME" --zone="$VM_ZONE" --project="$PROJECT"
            else
                echo "Cannot deploy to a stopped VM."
                exit 1
            fi
        fi
    fi

    # Confirm
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Deploy Summary"
    echo "════════════════════════════════════════════"
    echo ""
    echo "  VM:       $VM_NAME ($VM_ZONE)"
    echo "  Backend:  $BACKEND"
    echo "  Flags:    ${SETUP_FLAGS[*]:-none}"
    echo ""
    if [ "$AUTO_YES" = "false" ]; then
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        SETUP_FLAGS+=("--yes")
    fi

    # Wait for SSH
    wait_for_ssh "$VM_NAME" "$VM_ZONE"

    # Upload scripts
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ""
    echo "Uploading scripts..."

    case "$BACKEND" in
        llamacpp)
            scp_to_vm "$VM_NAME" "$VM_ZONE" "$SCRIPT_DIR/setup_llamacpp.sh"
            REMOTE_CMD="chmod +x ~/setup_llamacpp.sh && ~/setup_llamacpp.sh ${SETUP_FLAGS[*]}"
            ;;
        llamacpp-docker)
            scp_to_vm "$VM_NAME" "$VM_ZONE" --recurse "$SCRIPT_DIR/docker/"
            REMOTE_CMD="cd ~/docker && chmod +x setup_docker.sh && ./setup_docker.sh ${SETUP_FLAGS[*]}"
            ;;
        vllm)
            scp_to_vm "$VM_NAME" "$VM_ZONE" --recurse "$SCRIPT_DIR/vllm/"
            REMOTE_CMD="chmod +x ~/vllm/setup_vllm.sh && ~/vllm/setup_vllm.sh ${SETUP_FLAGS[*]}"
            ;;
    esac

    echo "  Upload complete."

    # Run setup
    echo ""
    echo "Running setup on $VM_NAME..."
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    GH_TOKEN_EXPORT=""
    if [ "$BACKEND" = "llamacpp-docker" ] && [ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]; then
        GH_TOKEN_EXPORT="export GH_TOKEN='${GH_TOKEN:-${GITHUB_TOKEN:-}}' && "
    fi

    SETUP_LOG="/tmp/deploy-setup.log"
    WRAPPED_CMD="${GH_TOKEN_EXPORT}$REMOTE_CMD 2>&1 | tee $SETUP_LOG; echo \"EXIT_CODE=\$?\" >> $SETUP_LOG"

    gcloud compute ssh "$VM_NAME" \
        --zone="$VM_ZONE" \
        --project="$PROJECT" \
        --ssh-flag="-o ServerAliveInterval=30" \
        --ssh-flag="-o ServerAliveCountMax=10" \
        --command="$WRAPPED_CMD" || {
        echo ""
        echo "SSH disconnected. The setup may still be running on the VM."
        echo ""
        echo "To check progress:"
        echo "  gcloud compute ssh $VM_NAME --zone=$VM_ZONE --project=$PROJECT --command='tail -f $SETUP_LOG'"
        exit 1
    }

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "Setup complete on $VM_NAME."

    # Print results
    EXTERNAL_IP=$(get_vm_ip "$VM_NAME" "$VM_ZONE")
    DEFAULT_PORT=8000
    case "$BACKEND" in
        llamacpp|llamacpp-docker) DEFAULT_PORT=8080 ;;
    esac

    echo ""
    echo "════════════════════════════════════════════"
    echo "  Deployment Complete"
    echo "════════════════════════════════════════════"
    echo ""
    echo "  VM:       $VM_NAME ($VM_ZONE)"
    echo "  Backend:  $BACKEND"
    echo "  IP:       ${EXTERNAL_IP:-(pending)}"
    echo "  API:      http://${EXTERNAL_IP}:${DEFAULT_PORT}/v1/"
    echo ""
    echo "  Creds:    ./llm.sh creds $VM_NAME"
    echo "  Stop:     ./llm.sh stop $VM_NAME"
    echo "  Resume:   ./llm.sh resume $VM_NAME"
    echo ""
}

# ===================================================================
# Main command dispatch
# ===================================================================
COMMAND="${1:-}"
shift 2>/dev/null || true

case "$COMMAND" in
    deploy)
        do_deploy "$@"
        ;;

    list|ls)
        gcloud compute instances list \
            --project="$PROJECT" \
            --format="table(name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP,machineType.machine_type().basename())"
        ;;

    stop)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        echo "Stopping $VM_NAME in $ZONE..."
        gcloud compute instances stop "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT"
        echo "Stopped. Disk preserved, GPU billing stopped."
        ;;

    start)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        echo "Starting $VM_NAME in $ZONE..."
        gcloud compute instances start "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT"
        IP=$(gcloud compute instances describe "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "(pending)")
        echo "Started. IP: $IP"
        echo ""
        echo "Restart service: ./llm.sh resume $VM_NAME"
        ;;

    resume)
        VM_NAME=""
        BACKEND=""
        VM_ZONE=""
        while [[ $# -gt 0 ]]; do
            case $1 in
                --backend) BACKEND="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
                --zone) VM_ZONE="$2"; shift 2 ;;
                -*) echo "Unknown option: $1"; exit 1 ;;
                *) VM_NAME="$1"; shift ;;
            esac
        done
        if [ -z "$VM_NAME" ]; then
            echo "Usage: ./llm.sh resume <vm-name> [--backend <name>]"
            exit 1
        fi
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")

        STATUS=$(get_vm_status "$VM_NAME" "$ZONE")
        if [ "$STATUS" != "RUNNING" ]; then
            echo "Starting $VM_NAME in $ZONE..."
            gcloud compute instances start "$VM_NAME" --zone="$ZONE" --project="$PROJECT"
        else
            echo "$VM_NAME is already running."
        fi

        wait_for_ssh "$VM_NAME" "$ZONE"

        if [ -z "$BACKEND" ]; then
            echo "Detecting backend..."
            BACKEND=$(detect_backend "$VM_NAME" "$ZONE")
            BACKEND=$(echo "$BACKEND" | tr -d '[:space:]')
            if [ -z "$BACKEND" ]; then
                echo "  Could not auto-detect backend. Please select one:"
                pick_backend
            else
                echo "  Detected: $BACKEND"
            fi
        fi

        do_deploy "$VM_NAME" --backend "$BACKEND" --start-only --yes --zone "$ZONE"
        ;;

    delete|rm)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        echo "WARNING: This will permanently delete $VM_NAME and its disk."
        read -p "Type the VM name to confirm: " confirm
        if [ "$confirm" != "$VM_NAME" ]; then
            echo "Cancelled."
            exit 0
        fi
        gcloud compute instances delete "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --quiet
        echo "Deleted."
        ;;

    ssh)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT"
        ;;

    ip)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        gcloud compute instances describe "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
        ;;

    creds|info)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        IP=$(gcloud compute instances describe "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
        CREDS=$(gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --command="
                KEY=\$(cat ~/qwen-*/.api_key 2>/dev/null || cat ~/llama-docker/.api_key 2>/dev/null || echo '(not found)')
                PORT=\$(ss -tlnp 2>/dev/null | grep -oP '0\.0\.0\.0:\K(8080|8000)' | head -1 || echo '8080')
                MODEL=\$(curl -s -H \"Authorization: Bearer \$KEY\" http://localhost:\$PORT/v1/models 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])' 2>/dev/null || echo '(unknown)')
                echo \"\$KEY|\$PORT|\$MODEL\"
            " 2>/dev/null)
        API_KEY=$(echo "$CREDS" | cut -d'|' -f1)
        PORT=$(echo "$CREDS" | cut -d'|' -f2)
        MODEL=$(echo "$CREDS" | cut -d'|' -f3)
        echo ""
        echo "  IP:       $IP"
        echo "  Port:     $PORT"
        echo "  Model:    $MODEL"
        echo "  API Key:  $API_KEY"
        echo "  Base URL: http://$IP:$PORT/v1/"
        echo ""
        ;;

    config)
        CONFIG_KEY=""
        CONFIG_VAL=""
        VM_NAME=""
        VM_ZONE=""

        while [[ $# -gt 0 ]]; do
            case $1 in
                --zone) VM_ZONE="$2"; shift 2 ;;
                -*) echo "Unknown option: $1"; exit 1 ;;
                *)
                    if [ -z "$VM_NAME" ]; then VM_NAME="$1"
                    elif [ -z "$CONFIG_KEY" ]; then CONFIG_KEY="$1"
                    elif [ -z "$CONFIG_VAL" ]; then CONFIG_VAL="$1"
                    fi
                    shift ;;
            esac
        done

        if [ -z "$VM_NAME" ] || [ -z "$CONFIG_KEY" ] || [ -z "$CONFIG_VAL" ]; then
            echo "Usage: ./llm.sh config <vm-name> <key> <value>"
            echo "Keys: context-length, parallel, port, thinking"
            exit 1
        fi

        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")

        case "$CONFIG_KEY" in
            context-length)
                SED_SYSTEMD="s/--ctx-size [0-9]*/--ctx-size $CONFIG_VAL/"
                SED_ENV="s/CONTEXT_LENGTH=.*/CONTEXT_LENGTH=$CONFIG_VAL/"
                ;;
            parallel)
                SED_SYSTEMD="s/--parallel [0-9]*/--parallel $CONFIG_VAL/"
                SED_ENV="s/PARALLEL=.*/PARALLEL=$CONFIG_VAL/"
                ;;
            port)
                SED_SYSTEMD="s/--port [0-9]*/--port $CONFIG_VAL/"
                SED_ENV="s/PORT=.*/PORT=$CONFIG_VAL/"
                ;;
            thinking)
                if [ "$CONFIG_VAL" = "on" ]; then
                    SED_SYSTEMD="/chat-template-kwargs/d"
                    SED_ENV=""
                else
                    echo "  Note: thinking is off by default. Use 'on' to enable."
                    exit 0
                fi
                ;;
            *)
                echo "Unknown config key: $CONFIG_KEY"
                echo "Available: context-length, parallel, port, thinking"
                exit 1
                ;;
        esac

        echo "Setting $CONFIG_KEY=$CONFIG_VAL on $VM_NAME..."

        gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --command="
                if [ -f /etc/systemd/system/llamacpp.service ]; then
                    sudo sed -i '$SED_SYSTEMD' /etc/systemd/system/llamacpp.service
                    sudo systemctl daemon-reload
                    sudo systemctl restart llamacpp.service
                    echo '  Updated llamacpp service and restarted.'
                elif [ -f ~/llama-docker/.env ]; then
                    sed -i '$SED_ENV' ~/llama-docker/.env
                    cd ~/llama-docker && sudo docker compose down && sudo docker compose up -d
                    echo '  Updated Docker config and restarted.'
                else
                    echo '  No known backend found on this VM.'
                fi
            " 2>/dev/null

        echo "  Done. Verify with: ./llm.sh creds $VM_NAME"
        ;;

    test)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        IP=$(gcloud compute instances describe "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)

        if [ -z "$IP" ]; then
            echo "ERROR: Could not get IP for $VM_NAME"
            exit 1
        fi

        CREDS=$(gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --command="
                KEY=\$(cat ~/qwen-*/.api_key 2>/dev/null || cat ~/llama-docker/.api_key 2>/dev/null || echo '')
                PORT=\$(ss -tlnp 2>/dev/null | grep -oP '0\.0\.0\.0:\K(8080|8000)' | head -1 || echo '8080')
                echo \"\$KEY|\$PORT\"
            " 2>/dev/null)
        API_KEY=$(echo "$CREDS" | cut -d'|' -f1)
        PORT=$(echo "$CREDS" | cut -d'|' -f2)
        BASE="http://$IP:$PORT"
        PASS=0
        FAIL=0

        echo ""
        echo "Testing $VM_NAME ($BASE)..."
        echo ""

        # 1. Health check (retries while model loads, up to 2 minutes)
        printf "  %-30s" "Health check..."
        HTTP_CODE="000"
        for i in $(seq 1 24); do
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BASE/health" 2>/dev/null || echo "000")
            if [ "$HTTP_CODE" = "200" ]; then break; fi
            if [ "$i" -eq 1 ]; then printf "\n    Waiting for model to load"; fi
            printf "."
            sleep 5
        done
        if [ "$HTTP_CODE" = "200" ]; then
            if [ "$i" -gt 1 ] 2>/dev/null; then printf " "; fi
            echo "PASS"
            PASS=$((PASS + 1))
        else
            if [ "$i" -gt 1 ] 2>/dev/null; then printf " "; fi
            echo "FAIL (HTTP $HTTP_CODE after ${i} attempts)"
            FAIL=$((FAIL + 1))
        fi

        # 2. List models
        printf "  %-30s" "List models..."
        MODELS_RESPONSE=$(curl -s --max-time 10 -H "Authorization: Bearer $API_KEY" "$BASE/v1/models" 2>/dev/null)
        MODEL=$(echo "$MODELS_RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
        if [ -n "$MODEL" ]; then
            echo "PASS ($MODEL)"
            PASS=$((PASS + 1))
        else
            echo "FAIL"
            FAIL=$((FAIL + 1))
            MODEL="qwen3.6-27b"
        fi

        # 3. Auth check (inference without key should fail)
        printf "  %-30s" "Auth (reject no key)..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
            -H "Content-Type: application/json" \
            "$BASE/v1/chat/completions" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}" \
            2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
            echo "PASS (HTTP $HTTP_CODE)"
            PASS=$((PASS + 1))
        elif [ "$HTTP_CODE" = "200" ]; then
            echo "WARN (no auth required)"
            PASS=$((PASS + 1))
        else
            echo "FAIL (HTTP $HTTP_CODE)"
            FAIL=$((FAIL + 1))
        fi

        # 4. Chat completion
        printf "  %-30s" "Chat completion..."
        CHAT_RESPONSE=$(curl -s --max-time 60 \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            "$BASE/v1/chat/completions" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in exactly 3 words.\"}],\"max_tokens\":20}" \
            2>/dev/null)
        CONTENT=$(echo "$CHAT_RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:80])' 2>/dev/null)
        if [ -n "$CONTENT" ]; then
            echo "PASS"
            echo "    → $CONTENT"
            PASS=$((PASS + 1))
        else
            ERROR=$(echo "$CHAT_RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("error",{}).get("message","unknown"))' 2>/dev/null || echo "no response")
            echo "FAIL ($ERROR)"
            FAIL=$((FAIL + 1))
        fi

        # 5. Streaming
        printf "  %-30s" "Streaming..."
        STREAM_CHUNKS=$(curl -s --max-time 60 \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            "$BASE/v1/chat/completions" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":10,\"stream\":true}" \
            2>/dev/null | grep -c "^data:" || echo "0")
        if [ "$STREAM_CHUNKS" -gt 1 ] 2>/dev/null; then
            echo "PASS ($STREAM_CHUNKS chunks)"
            PASS=$((PASS + 1))
        else
            echo "FAIL ($STREAM_CHUNKS chunks)"
            FAIL=$((FAIL + 1))
        fi

        # Summary
        echo ""
        TOTAL=$((PASS + FAIL))
        if [ "$FAIL" -eq 0 ]; then
            echo "  All $TOTAL tests passed."
        else
            echo "  $PASS/$TOTAL passed, $FAIL failed."
        fi
        echo ""

        [ "$FAIL" -eq 0 ] || exit 1
        ;;

    logs)
        parse_vm_args "$@"
        ZONE=$(resolve_zone "$VM_NAME" "$VM_ZONE")
        gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            --project="$PROJECT" \
            --command="sudo journalctl -u llamacpp.service -u 'vllm-*.service' -n 50 --no-pager 2>/dev/null"
        ;;

    ""|--help|-h)
        show_usage
        ;;

    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
