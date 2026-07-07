#!/bin/bash

# Shared ngrok tunnel helpers for llm-bootstrap setup scripts.
# Uploaded to ~/ngrok.sh on the VM by llm.sh and sourced from setup_llamacpp.sh,
# docker/setup_docker.sh, and vllm/setup_vllm.sh when --ngrok is passed.

ngrok_ensure_installed() {
    if command -v ngrok &>/dev/null; then
        return 0
    fi
    echo "  Installing ngrok..."
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
        | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq ngrok
}

# ngrok_resolve_authtoken <auto_yes>
# Echoes the resolved token to stdout (env var, or interactive prompt).
# Returns 1 with nothing echoed if no token could be resolved.
ngrok_resolve_authtoken() {
    local auto_yes="$1"

    if [ -n "$NGROK_AUTHTOKEN" ]; then
        echo "$NGROK_AUTHTOKEN"
        return 0
    fi

    if [ "$auto_yes" = "true" ]; then
        return 1
    fi

    local token_input
    read -p "Enter your ngrok authtoken (https://dashboard.ngrok.com/get-started/your-authtoken): " token_input
    if [ -z "$token_input" ]; then
        return 1
    fi
    echo "$token_input"
}

# ngrok_start_tunnel <port> <authtoken>
# Starts (or restarts) a systemd-managed ngrok tunnel for the given port,
# writes the resulting public URL to ~/.ngrok_url, and prints it.
ngrok_start_tunnel() {
    local port="$1"
    local authtoken="$2"

    if [ -z "$authtoken" ]; then
        echo "ERROR: No ngrok authtoken provided (set NGROK_AUTHTOKEN)." >&2
        return 1
    fi

    ngrok config add-authtoken "$authtoken"

    sudo tee /etc/systemd/system/ngrok.service > /dev/null << EOF
[Unit]
Description=ngrok tunnel
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$(command -v ngrok) http $port --log=stdout
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ngrok.service
    sudo systemctl restart ngrok.service

    echo "  Waiting for ngrok tunnel..."
    local url=""
    for i in $(seq 1 15); do
        url=$(curl -s http://localhost:4040/api/tunnels \
            | python3 -c 'import sys, json
tunnels = json.load(sys.stdin)["tunnels"]
print(tunnels[0]["public_url"] if tunnels else "")' 2>/dev/null)
        [ -n "$url" ] && break
        sleep 2
    done

    if [ -z "$url" ]; then
        echo "  WARNING: Could not retrieve ngrok URL. Check: sudo journalctl -u ngrok.service"
        return 1
    fi

    echo "$url" > "$HOME/.ngrok_url"
    echo "  ngrok tunnel ready: $url"
    echo "  Note: this URL changes on every tunnel restart unless you use a reserved ngrok domain."
}

# ngrok_setup_if_enabled <enabled> <port> <auto_yes>
# One-shot helper for setup scripts: no-op unless <enabled> is "true".
ngrok_setup_if_enabled() {
    local enabled="$1" port="$2" auto_yes="$3"
    [ "$enabled" = "true" ] || return 0

    echo "Setting up ngrok tunnel..."
    ngrok_ensure_installed

    local authtoken
    authtoken=$(ngrok_resolve_authtoken "$auto_yes") || {
        echo "  WARNING: No ngrok authtoken available, skipping tunnel. Set NGROK_AUTHTOKEN and re-run with --ngrok." >&2
        return 1
    }

    ngrok_start_tunnel "$port" "$authtoken"
}
