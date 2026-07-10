#!/bin/bash

# Shared helper for self-converting the DFlash speculative-decoding draft
# model from its primary source, instead of trusting a pre-built third-party
# GGUF. Uploaded to ~/dflash_convert.sh on the VM by llm.sh and sourced from
# setup_llamacpp.sh and docker/setup_docker.sh when --dflash is passed.
#
# Why self-convert: every pre-built DFlash GGUF on Hugging Face is someone
# else's conversion, and several popular ones predate llama.cpp PR #22105's
# architecture rename (dflash-draft -> dflash), silently failing to load.
# Converting from z-lab's original MIT-licensed safetensors release sidesteps
# that entirely — see docs/dflash-research.md for the full writeup.
DFLASH_SRC_REPO="z-lab/Qwen3.6-27B-DFlash"
DFLASH_TARGET_REPO="Qwen/Qwen3.6-27B"

# convert_dflash_draft <llama_cpp_src_dir> <dest_gguf_path> <marker_path>
# Downloads the DFlash draft safetensors + the target's tokenizer, converts
# to GGUF via llama.cpp's own convert_hf_to_gguf.py, and caches the result.
convert_dflash_draft() {
    local llama_src="$1"
    local dest="$2"
    local marker="$3"

    if [ -f "$dest" ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$DFLASH_SRC_REPO" ]; then
        echo "  DFlash draft already converted."
        return 0
    fi

    if [ ! -f "$llama_src/convert_hf_to_gguf.py" ]; then
        echo "  ERROR: convert_hf_to_gguf.py not found in $llama_src" >&2
        return 1
    fi

    echo "  Converting DFlash draft from primary source (one-time, a few minutes)..."

    echo "    Installing conversion dependencies..."
    pip3 install --quiet -r "$llama_src/requirements/requirements-convert_hf_to_gguf.txt" 2>/dev/null || \
        pip3 install --quiet --break-system-packages -r "$llama_src/requirements/requirements-convert_hf_to_gguf.txt" || {
        echo "  ERROR: Failed to install llama.cpp conversion dependencies. Try manually:" >&2
        echo "    pip3 install -r $llama_src/requirements/requirements-convert_hf_to_gguf.txt" >&2
        return 1
    }

    local hf_cmd="hf"
    command -v hf &>/dev/null || hf_cmd="huggingface-cli"

    local scratch
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' RETURN

    echo "    Downloading $DFLASH_SRC_REPO (draft weights)..."
    $hf_cmd download "$DFLASH_SRC_REPO" --local-dir "$scratch/dflash-src" || {
        echo "  ERROR: Failed to download $DFLASH_SRC_REPO" >&2
        return 1
    }

    echo "    Downloading tokenizer from $DFLASH_TARGET_REPO (config/tokenizer files only, not model weights)..."
    $hf_cmd download "$DFLASH_TARGET_REPO" \
        --include "*.json" --include "tokenizer*" --exclude "*.safetensors" \
        --local-dir "$scratch/target-tokenizer" || {
        echo "  ERROR: Failed to download tokenizer from $DFLASH_TARGET_REPO" >&2
        return 1
    }

    echo "    Running convert_hf_to_gguf.py..."
    mkdir -p "$(dirname "$dest")"
    python3 "$llama_src/convert_hf_to_gguf.py" "$scratch/dflash-src" \
        --target-model-dir "$scratch/target-tokenizer" \
        --outtype bf16 --outfile "$dest" || {
        echo "  ERROR: DFlash conversion failed." >&2
        return 1
    }

    echo "$DFLASH_SRC_REPO" > "$marker"
    echo "  DFlash draft converted: $dest"
}
