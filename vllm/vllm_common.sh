#!/bin/bash

# Common library for vLLM setup scripts
# Source this file in your setup scripts with: source vllm_common.sh

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Tool calling parser mapping
declare -A TOOL_CALL_PARSERS
TOOL_CALL_PARSERS["gpt-oss"]="hermes"     # Harmony format works with hermes
TOOL_CALL_PARSERS["qwen"]="hermes"        # Use hermes for Qwen (qwen parser has bugs)
TOOL_CALL_PARSERS["llama"]="llama3_json"
TOOL_CALL_PARSERS["mistral"]="mistral"
TOOL_CALL_PARSERS["deepseek"]="deepseek_v3"

get_tool_call_parser() {
    local model_name=$1

    # Detect model family from model name
    if [[ "$model_name" == *"gpt-oss"* ]]; then
        echo "hermes"  # Harmony format
    elif [[ "$model_name" == *"Qwen3.5"* ]] || [[ "$model_name" == *"qwen3.5"* ]]; then
        echo "qwen3_coder"  # Qwen 3.5 uses qwen3_coder parser
    elif [[ "$model_name" == *"Qwen"* ]] || [[ "$model_name" == *"qwen"* ]]; then
        echo "hermes"  # Older Qwen models use hermes (qwen parser has bugs)
    elif [[ "$model_name" == *"Llama"* ]] || [[ "$model_name" == *"llama"* ]]; then
        echo "llama3_json"
    elif [[ "$model_name" == *"Mistral"* ]] || [[ "$model_name" == *"mistral"* ]]; then
        echo "mistral"
    else
        echo "hermes"  # Default to hermes
    fi
}

get_mcp_compatibility_level() {
    local model_name=$1

    if [[ "$model_name" == *"gpt-oss"* ]]; then
        echo "⭐⭐⭐⭐⭐ Excellent (Harmony format, recommended for Claude Code)"
    elif [[ "$model_name" == *"Qwen"* ]] || [[ "$model_name" == *"qwen"* ]]; then
        echo "⭐⭐⭐ Limited (XML format, known compatibility issues)"
    else
        echo "⭐⭐⭐⭐ Good (standard JSON format)"
    fi
}

# Print functions
print_header() {
    echo -e "${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# System setup functions
update_system() {
    print_info "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    print_success "System updated"
}

install_essential_packages() {
    print_info "Installing essential packages..."
    sudo apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        tmux \
        htop \
        nvtop 2>/dev/null || true
    print_success "Essential packages installed"
}

install_uv() {
    print_info "Installing uv package manager..."

    # Check if uv is already installed
    if command -v uv &> /dev/null; then
        print_success "uv already installed: $(uv --version)"
        return 0
    fi

    # Install uv using official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session (uv installs to ~/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"

    # Verify installation
    if command -v uv &> /dev/null; then
        print_success "uv installed: $(uv --version)"
    else
        print_error "uv installation failed"
        exit 1
    fi
}

verify_nvidia_drivers() {
    print_info "Checking NVIDIA GPU drivers..."

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found!"
        echo ""
        echo "Please install NVIDIA drivers first:"
        echo "  sudo /opt/deeplearning/install-driver.sh"
        echo ""
        echo "Or manually install drivers for your GPU."
        exit 1
    fi

    nvidia-smi
    print_success "NVIDIA drivers found"
}

check_gpu_memory() {
    local required_memory=$1
    local gpu_count=$2

    print_info "Checking GPU memory requirements..."

    # Get total GPU memory across all GPUs
    local total_memory=0
    local gpu_memories=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

    while IFS= read -r mem; do
        total_memory=$((total_memory + mem))
    done <<< "$gpu_memories"

    local num_gpus=$(echo "$gpu_memories" | wc -l)

    echo "  Detected: $num_gpus GPU(s) with ${total_memory}MB total memory"
    echo "  Required: $gpu_count GPU(s) with ${required_memory}MB total memory"

    if [ "$num_gpus" -lt "$gpu_count" ]; then
        print_warning "Only $num_gpus GPU(s) detected, but $gpu_count recommended"
        if [ "${AUTO_YES:-false}" = "false" ]; then
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_warning "Auto-continuing despite GPU count warning (--yes flag)"
        fi
    fi

    if [ "$total_memory" -lt "$required_memory" ]; then
        print_warning "Total GPU memory ($total_memory MB) is less than recommended ($required_memory MB)"
        if [ "${AUTO_YES:-false}" = "false" ]; then
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            print_warning "Auto-continuing despite GPU memory warning (--yes flag)"
        fi
    else
        print_success "GPU memory check passed"
    fi
}

create_venv() {
    local work_dir=$1

    print_info "Creating Python virtual environment..."
    mkdir -p "$work_dir"
    cd "$work_dir"
    python3.10 -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created"
}

install_pytorch() {
    print_info "Installing PyTorch with CUDA 12.1 using uv..."
    uv pip install --upgrade pip
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    print_success "PyTorch installed"
}

install_vllm() {
    local use_nightly=${1:-false}

    if [ "$use_nightly" = "true" ]; then
        print_info "Installing vLLM nightly using uv (required for Qwen 3.5 models)..."
        print_warning "Using official vLLM nightly installation method"
        # Official vLLM Qwen 3.5 installation command
        uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
        print_success "vLLM nightly installed"
    else
        print_info "Installing vLLM stable using uv..."
        uv pip install vllm
        print_success "vLLM stable installed"
    fi
}

install_hf_dependencies() {
    print_info "Installing Hugging Face dependencies using uv..."
    uv pip install huggingface-hub transformers accelerate
    print_success "Hugging Face dependencies installed"
}

huggingface_login() {
    echo ""
    print_header "Hugging Face Authentication"

    # Check if HF_TOKEN is already set in environment
    if [ -n "$HF_TOKEN" ]; then
        print_info "Using HF_TOKEN from environment"
        uv pip install -U "huggingface_hub[cli]"
        huggingface-cli login --token "$HF_TOKEN"
        print_success "Logged in to Hugging Face"
        return
    fi

    # Skip prompt if running non-interactively (AUTO_YES flag)
    if [ "${AUTO_YES:-false}" = "true" ]; then
        print_info "Auto-skipped Hugging Face login (--yes flag, model is public)"
        return
    fi

    # Interactive prompt
    echo "If the model requires authentication, please enter your HF token."
    echo "You can skip this if the model is public."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    read -p "Enter Hugging Face token (or press Enter to skip): " HF_TOKEN_INPUT

    if [ -n "$HF_TOKEN_INPUT" ]; then
        uv pip install -U "huggingface_hub[cli]"
        huggingface-cli login --token "$HF_TOKEN_INPUT"
        print_success "Logged in to Hugging Face"
    else
        print_info "Skipped Hugging Face login"
    fi
}

download_model() {
    local model_name=$1
    local model_size=$2

    print_info "Pre-downloading model: $model_name"
    print_info "Expected size: ~$model_size"
    print_warning "This may take a while depending on your connection..."

    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$model_name', local_files_only=False)"
    print_success "Model downloaded"
}

generate_api_key() {
    local work_dir=$1

    local key_file="$work_dir/.api_key"
    if [ -f "$key_file" ]; then
        print_info "API key already exists at $key_file"
    else
        openssl rand -hex 32 > "$key_file"
        chmod 600 "$key_file"
        print_success "API key generated at $key_file"
    fi
    cat "$key_file"
}

create_vllm_start_script() {
    local work_dir=$1
    local model_name=$2
    local tensor_parallel=$3
    local max_model_len=$4
    local gpu_memory_util=$5
    local served_model_name=$6
    local extra_args=$7
    local enable_tool_calling=${8:-false}
    local cuda_visible_devices=${9:-""}
    local api_key_file=${10:-""}
    local port=${11:-8000}

    print_info "Creating vLLM server startup script..."

    # Detect reasoning parser for Qwen 3.5 models
    local reasoning_parser=""
    if [[ "$model_name" == *"Qwen3.5"* ]] || [[ "$model_name" == *"qwen3.5"* ]]; then
        reasoning_parser="--reasoning-parser qwen3"
        print_info "Detected Qwen 3.5 model - adding reasoning parser"
    fi

    # Configure tool calling if enabled
    local tool_call_args=""
    if [ "$enable_tool_calling" = "true" ]; then
        local parser=$(get_tool_call_parser "$model_name")
        tool_call_args="--enable-auto-tool-choice --tool-call-parser $parser"
        print_info "Tool calling enabled with parser: $parser"
        local compat=$(get_mcp_compatibility_level "$model_name")
        print_info "MCP/Claude Code compatibility: $compat"
    fi

    # Prepare CUDA_VISIBLE_DEVICES export if specified
    local cuda_export=""
    if [ -n "$cuda_visible_devices" ]; then
        cuda_export="export CUDA_VISIBLE_DEVICES=$cuda_visible_devices"
        print_info "Constraining to GPUs: $cuda_visible_devices"
    fi

    # API key auth
    local api_key_arg=""
    if [ -n "$api_key_file" ]; then
        api_key_arg="--api-key \$(cat $api_key_file)"
        print_info "API key auth enabled via $api_key_file"
    fi

    cat > "$work_dir/start_vllm_server.sh" << EOF
#!/bin/bash

# Constrain GPU visibility if specified
$cuda_export

# Activate virtual environment
source $work_dir/venv/bin/activate

# Start vLLM server
vllm serve $model_name \\
    --host 0.0.0.0 \\
    --port $port \\
    --tensor-parallel-size $tensor_parallel \\
    --max-model-len $max_model_len \\
    --gpu-memory-utilization $gpu_memory_util \\
    --enable-prefix-caching \\
    --max-num-seqs 256 \\
    --served-model-name $served_model_name \\
    --trust-remote-code \\
    --dtype auto \\
    $reasoning_parser \\
    $tool_call_args \\
    $api_key_arg \\
    $extra_args

EOF

    chmod +x "$work_dir/start_vllm_server.sh"
    print_success "Start script created at $work_dir/start_vllm_server.sh"
}

create_systemd_service() {
    local service_name=$1
    local work_dir=$2
    local description=$3

    print_info "Creating systemd service: $service_name"

    sudo tee /etc/systemd/system/$service_name.service > /dev/null << EOF
[Unit]
Description=$description
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$work_dir
ExecStart=$work_dir/start_vllm_server.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable $service_name.service
    print_success "Systemd service created and enabled"
}

configure_firewall() {
    local port=$1

    print_info "Configuring firewall for port $port..."

    if command -v ufw &> /dev/null; then
        sudo ufw allow $port/tcp
        print_success "UFW firewall configured"
    else
        print_info "UFW not found, skipping firewall configuration"
    fi
}

create_test_script() {
    local work_dir=$1
    local model_name=$2
    local display_name=$3

    print_info "Creating API test script..."

    cat > "$work_dir/test_api.py" << 'EOFPYTHON'
#!/usr/bin/env python3
import requests
import json
import time
import sys

MODEL_NAME = "MODEL_PLACEHOLDER"
DISPLAY_NAME = "DISPLAY_PLACEHOLDER"

url = "http://localhost:8000/v1/chat/completions"
passed = 0
failed = 0

def pass_test(msg=""):
    global passed
    passed += 1
    print(f"  PASS{': ' + msg if msg else ''}")

def fail_test(msg):
    global failed
    failed += 1
    print(f"  FAIL: {msg}")

print(f"Testing {DISPLAY_NAME} API...")
print("=" * 60)

# Test 1: Simple response
print("\n1. Chat completion...")
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "What is 2+2? Answer in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 50
}

try:
    start_time = time.time()
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    content = result['choices'][0]['message']['content']
    elapsed = time.time() - start_time
    assert content, "Empty response"
    print(f"  Answer: {content}")
    print(f"  Tokens: {result['usage']}")
    pass_test(f"{elapsed:.2f}s")
except Exception as e:
    fail_test(str(e))

# Test 2: Code generation
print("\n2. Code generation...")
payload_code = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": "Write a Python function to reverse a string. Just the code, no explanation."}
    ],
    "temperature": 0.6,
    "max_tokens": 200
}

try:
    start_time = time.time()
    response = requests.post(url, json=payload_code, timeout=120)
    response.raise_for_status()
    result = response.json()
    content = result['choices'][0]['message']['content']
    elapsed = time.time() - start_time
    assert content, "Empty response"
    print(f"  {content[:120]}...")
    pass_test(f"{elapsed:.2f}s")
except Exception as e:
    fail_test(str(e))

# Test 3: Streaming
print("\n3. Streaming...")
payload_stream = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Count from 1 to 5 with brief descriptions."}
    ],
    "temperature": 0.7,
    "max_tokens": 150,
    "stream": True
}

try:
    response = requests.post(url, json=payload_stream, stream=True, timeout=120)
    response.raise_for_status()
    chunks = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: ') and line != 'data: [DONE]':
                data = json.loads(line[6:])
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        chunks += 1
    assert chunks > 0, "No chunks received"
    pass_test(f"{chunks} chunks")
except Exception as e:
    fail_test(str(e))

# Test 4: List models
print("\n4. List models...")
try:
    response = requests.get("http://localhost:8000/v1/models", timeout=10)
    response.raise_for_status()
    models = response.json()
    model_ids = [m['id'] for m in models.get('data', [])]
    assert model_ids, "No models returned"
    print(f"  Models: {', '.join(model_ids)}")
    pass_test()
except Exception as e:
    fail_test(str(e))

# Results
print("\n" + "=" * 60)
total = passed + failed
if failed == 0:
    print(f"All {total} tests passed.")
else:
    print(f"{passed}/{total} passed, {failed} failed.")
    sys.exit(1)

EOFPYTHON

    # Replace placeholders
    sed -i "s/MODEL_PLACEHOLDER/$model_name/g" "$work_dir/test_api.py"
    sed -i "s/DISPLAY_PLACEHOLDER/$display_name/g" "$work_dir/test_api.py"

    chmod +x "$work_dir/test_api.py"
    print_success "Test script created at $work_dir/test_api.py"
}

print_completion_message() {
    local service_name=$1
    local work_dir=$2
    local model_display_name=$3
    local served_model_name=$4
    local cost_estimate=$5

    echo ""
    print_header "Installation Complete!"
    echo ""
    echo "Model: $model_display_name"
    echo "Service: $service_name"
    echo "Working Directory: $work_dir"
    echo "Estimated Cost: $cost_estimate"
    echo ""
    print_info "Next steps:"
    echo ""
    echo "1. Start the vLLM server:"
    echo "   sudo systemctl start $service_name.service"
    echo ""
    echo "2. Check server status:"
    echo "   sudo systemctl status $service_name.service"
    echo ""
    echo "3. View logs (wait for 'Uvicorn running on http://0.0.0.0:8000'):"
    echo "   sudo journalctl -u $service_name.service -f"
    echo ""
    echo "4. Test the API (after server starts):"
    echo "   cd $work_dir && ./test_api.py"
    echo ""
    echo "5. Get your VM's external IP:"
    echo "   curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
    echo ""
    echo "6. Configure Claude Code MCP:"
    echo '   Add to your MCP config:'
    echo '   {'
    echo '     "mcpServers": {'
    echo '       "vllm": {'
    echo '         "command": "npx",'
    echo '         "args": ["-y", "@modelcontextprotocol/server-openai-compatible",'
    echo '                  "--base-url", "http://YOUR_VM_IP:8000/v1",'
    echo "                  \"--model\", \"$served_model_name\"]"
    echo '       }'
    echo '     }'
    echo '   }'
    echo ""
    print_warning "IMPORTANT: Configure GCP firewall to allow port 8000!"
    echo "   gcloud compute firewall-rules create allow-vllm \\"
    echo "     --allow=tcp:8000 \\"
    echo "     --source-ranges=YOUR_IP/32"
    echo ""
    print_success "Setup complete! Happy coding! 🚀"
    echo ""
}

# GPU memory size mappings (in MB)
declare -A GPU_MEMORY
GPU_MEMORY["t4"]=16384
GPU_MEMORY["a10g"]=24576
GPU_MEMORY["a100-40gb"]=40960
GPU_MEMORY["a100-80gb"]=81920
GPU_MEMORY["h100-80gb"]=81920
GPU_MEMORY["l4"]=24576
GPU_MEMORY["v100"]=16384

get_gpu_memory_estimate() {
    local gpu_type=$1
    echo "${GPU_MEMORY[$gpu_type]:-40960}"  # Default to 40GB if unknown
}
