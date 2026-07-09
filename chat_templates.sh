#!/bin/bash

# Shared helper for downloading community-fixed chat templates.
# Uploaded to ~/chat_templates.sh on the VM by llm.sh and sourced from
# setup_llamacpp.sh, docker/setup_docker.sh, and vllm/setup_vllm.sh when
# --fixed-chat-template is passed.
#
# Pinned to a specific commit (not "main") so the template used by a running
# server can't change out from under it if the upstream repo is edited later.
# Bump this hash deliberately when you want to pick up an update:
# https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
QWEN_FIXED_TEMPLATE_URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/23a40b0bd4d197c31d39e3c442fd2cd6100b3971/chat_template.jinja"

# download_fixed_chat_template <dest_path>
# Downloads the pinned Qwen 3.5/3.6 chat template fix to <dest_path>.
download_fixed_chat_template() {
    local dest="$1"
    echo "  Downloading fixed Qwen chat template..."
    if ! curl -sL --fail -o "$dest" "$QWEN_FIXED_TEMPLATE_URL"; then
        echo "  WARNING: Could not download fixed chat template, falling back to the model's default." >&2
        return 1
    fi
}
