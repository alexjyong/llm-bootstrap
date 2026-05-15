#!/bin/bash

# Loop through GCP zones to find GPU availability and create a VM.
# GCP has no real-time capacity API — you only discover a zone is full
# when the create call fails with ZONE_RESOURCE_POOL_EXHAUSTED.

set -euo pipefail

PROJECT="${GCP_PROJECT:-your-project-id}"
VM_NAME="llm-$(openssl rand -hex 3)"
DISK_SIZE=250
IMAGE="projects/ml-images/global/images/common-cu129-ubuntu-2204-nvidia-580-v20260430"
SUBNET="${GCP_SUBNET:-default}"
PROVISIONING="STANDARD"
GPU_PRESET=""
TRY_ZONE=""
NAME_SET=false
STATIC_IP=false
AUTO_STOP="4h"

show_usage() {
    cat << 'EOF'
Create a GCP GPU VM, looping through zones until one has capacity.

Usage: ./create_gpu_vm.sh [options]

Options:
  --gpu <l4|a100|a100x2>  GPU preset (default: l4)
  --name <name>           VM name (default: llm-<random>)
  --spot                  Use spot/preemptible pricing (~70% cheaper, can be preempted)
  --static-ip             Reserve a static external IP (persists across stop/start)
  --auto-stop <duration>  Auto-stop after duration (default: 4h). Examples: 4h, 12h, 1d
  --no-auto-stop          Disable auto-stop (VM runs until manually stopped)
  --zone <zone>           Try this zone first before looping
  --help, -h              Show this help

GPU presets:
  l4      2x L4 (48GB) — g2-standard-24
  a100    1x A100 (80GB) — a2-highgpu-1g
  a100x2  2x A100 (160GB) — a2-highgpu-2g

Examples:
  ./create_gpu_vm.sh                              # 2x L4, loop all zones
  ./create_gpu_vm.sh --gpu a100                   # 1x A100
  ./create_gpu_vm.sh --gpu a100x2                 # 2x A100
  ./create_gpu_vm.sh --spot                       # 2x L4, spot pricing
  ./create_gpu_vm.sh --zone us-east1-b            # try specific zone first
  ./create_gpu_vm.sh --gpu a100 --name dev --spot # A100 spot, custom name

After creation, deploy a backend:
  ./llm.sh deploy VM_NAME --backend llamacpp-docker --model 1 --quant Q6_K --yes
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --spot)
            PROVISIONING="SPOT"
            shift
            ;;
        --name)
            VM_NAME="$2"
            NAME_SET=true
            shift 2
            ;;
        --zone)
            TRY_ZONE="$2"
            shift 2
            ;;
        --gpu)
            GPU_PRESET="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --static-ip)
            STATIC_IP=true
            shift
            ;;
        --auto-stop)
            AUTO_STOP="$2"
            shift 2
            ;;
        --no-auto-stop)
            AUTO_STOP=""
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

# ───────────────────────────────────────────────────────────
# Interactive GPU picker (when no --gpu flag)
# ───────────────────────────────────────────────────────────
if [ -z "$GPU_PRESET" ]; then
    echo ""
    echo "Select GPU preset:"
    echo ""
    echo "  1) 2x L4 (48GB) — g2-standard-24 (default)"
    echo "     GGUF quants (Q4-Q8). Good context window, best value."
    echo ""
    echo "  2) 1x A100 (80GB) — a2-highgpu-1g"
    echo "     Fast inference (2TB/s bandwidth). Large context + high quants."
    echo ""
    echo "  3) 2x A100 (160GB) — a2-highgpu-2g"
    echo "     Full precision or very large models. vLLM multi-user serving."
    echo ""
    while true; do
        read -p "GPU preset [1-3] (Enter for default): " choice
        case "$choice" in
            ""|1) GPU_PRESET="l4"; break ;;
            2) GPU_PRESET="a100"; break ;;
            3) GPU_PRESET="a100x2"; break ;;
            *) echo "  Invalid choice. Enter 1-3 or press Enter for default." ;;
        esac
    done

    # Ask about static IP
    echo ""
    read -p "Reserve a static IP? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        STATIC_IP=true
    fi

    # Ask for VM name
    if [ "$NAME_SET" = "false" ]; then
        echo ""
        read -p "VM name (Enter for '$VM_NAME'): " custom_name
        if [ -n "$custom_name" ]; then
            VM_NAME="$custom_name"
        fi
    fi
fi

# ───────────────────────────────────────────────────────────
# Resolve GPU preset
# ───────────────────────────────────────────────────────────
case "$GPU_PRESET" in
    l4)
        MACHINE_TYPE="g2-standard-24"
        GPU_TYPE="nvidia-l4"
        GPU_COUNT=2
        GPU_LABEL="2x L4 (48GB)"
        ;;
    a100)
        MACHINE_TYPE="a2-highgpu-1g"
        GPU_TYPE="nvidia-tesla-a100"
        GPU_COUNT=1
        GPU_LABEL="1x A100 (80GB)"
        ;;
    a100x2)
        MACHINE_TYPE="a2-highgpu-2g"
        GPU_TYPE="nvidia-tesla-a100"
        GPU_COUNT=2
        GPU_LABEL="2x A100 (160GB)"
        ;;
    *)
        echo "ERROR: Unknown GPU preset '${GPU_PRESET}'"
        echo "Available: l4, a100, a100x2"
        exit 1
        ;;
esac

# ───────────────────────────────────────────────────────────
# Find zones with the selected GPU
# ───────────────────────────────────────────────────────────
echo "Finding zones with ${GPU_TYPE} GPUs..."
ALL_ZONES=$(gcloud compute accelerator-types list \
    --project="$PROJECT" \
    --filter="name:${GPU_TYPE}" \
    --format="value(zone)" \
    2>/dev/null | sort -u)

if [ -z "$ALL_ZONES" ]; then
    echo "ERROR: No zones found with ${GPU_TYPE} GPUs in project ${PROJECT}"
    exit 1
fi

ZONE_COUNT=$(echo "$ALL_ZONES" | wc -l)
echo "Found ${ZONE_COUNT} zones with ${GPU_TYPE}."

# Order: user-specified zone first, then US zones, then everything else
ORDERED_ZONES=""
if [ -n "$TRY_ZONE" ]; then
    ORDERED_ZONES="$TRY_ZONE"
fi
US_ZONES=$(echo "$ALL_ZONES" | grep "^us-" || true)
OTHER_ZONES=$(echo "$ALL_ZONES" | grep -v "^us-" || true)
for z in $US_ZONES $OTHER_ZONES; do
    if [ "$z" != "$TRY_ZONE" ]; then
        ORDERED_ZONES="${ORDERED_ZONES:+$ORDERED_ZONES }$z"
    fi
done

echo ""
echo "  VM name:   ${VM_NAME}"
echo "  Preset:    ${GPU_PRESET}"
echo "  Machine:   ${MACHINE_TYPE} (${GPU_LABEL})"
echo "  Pricing:   ${PROVISIONING}"
echo "  Auto-stop: $([ -n "$AUTO_STOP" ] && echo "after ${AUTO_STOP}" || echo "disabled")"
echo "  Disk:      ${DISK_SIZE}GB pd-balanced"
echo ""

# ───────────────────────────────────────────────────────────
# Try each zone
# ───────────────────────────────────────────────────────────
TRIED=0
for ZONE in $ORDERED_ZONES; do
    TRIED=$((TRIED + 1))
    REGION="${ZONE%-*}"
    echo "[${TRIED}/${ZONE_COUNT}] Trying ${ZONE}..."

    SPOT_FLAGS=""
    if [ "$PROVISIONING" = "SPOT" ]; then
        SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
    else
        SPOT_FLAGS="--provisioning-model=STANDARD"
    fi

    AUTO_STOP_FLAGS=""
    if [ -n "$AUTO_STOP" ]; then
        AUTO_STOP_FLAGS="--max-run-duration=${AUTO_STOP} --instance-termination-action=STOP"
    fi

    OUTPUT=$(gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --network-interface="network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=${SUBNET}" \
        --maintenance-policy=TERMINATE \
        $SPOT_FLAGS \
        $AUTO_STOP_FLAGS \
        --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write \
        --accelerator="count=${GPU_COUNT},type=${GPU_TYPE}" \
        --create-disk="auto-delete=yes,boot=yes,device-name=${VM_NAME},image=${IMAGE},mode=rw,size=${DISK_SIZE},type=pd-balanced" \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=goog-ec-src=vm_add-gcloud \
        --reservation-affinity=any \
        2>&1) && VM_CREATED=true || VM_CREATED=false

    if [ "$VM_CREATED" = "true" ]; then
        echo "$OUTPUT"
        echo ""
        echo "VM created in ${ZONE}!"

        # Set up snapshot schedule (may already exist in this region)
        echo "Setting up daily snapshot schedule in ${REGION}..."
        gcloud compute resource-policies create snapshot-schedule default-schedule-1 \
            --project="$PROJECT" \
            --region="$REGION" \
            --max-retention-days=14 \
            --on-source-disk-delete=keep-auto-snapshots \
            --daily-schedule \
            --start-time=03:00 2>/dev/null || true

        gcloud compute disks add-resource-policies "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --resource-policies="projects/${PROJECT}/regions/${REGION}/resourcePolicies/default-schedule-1" \
            2>/dev/null || true

        # Reserve static IP if requested
        if [ "$STATIC_IP" = "true" ]; then
            echo "Reserving static IP..."
            gcloud compute addresses create "${VM_NAME}-ip" \
                --project="$PROJECT" \
                --region="$REGION" 2>/dev/null || true

            STATIC_ADDR=$(gcloud compute addresses describe "${VM_NAME}-ip" \
                --project="$PROJECT" \
                --region="$REGION" \
                --format="get(address)" 2>/dev/null)

            if [ -n "$STATIC_ADDR" ]; then
                gcloud compute instances delete-access-config "$VM_NAME" \
                    --zone="$ZONE" \
                    --project="$PROJECT" \
                    --access-config-name="external-nat" 2>/dev/null || true
                gcloud compute instances add-access-config "$VM_NAME" \
                    --zone="$ZONE" \
                    --project="$PROJECT" \
                    --address="$STATIC_ADDR" 2>/dev/null
                echo "  Static IP reserved: $STATIC_ADDR"
            fi
        fi

        EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --format="get(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "(pending)")

        echo ""
        echo "════════════════════════════════════════════"
        echo "  VM created successfully"
        echo "════════════════════════════════════════════"
        echo ""
        echo "  Name:     ${VM_NAME}"
        echo "  Zone:     ${ZONE}"
        echo "  Type:     ${MACHINE_TYPE} (${GPU_LABEL})"
        echo "  Pricing:  ${PROVISIONING}"
        echo "  IP:       ${EXTERNAL_IP}$([ "$STATIC_IP" = "true" ] && echo " (static)" || echo " (ephemeral)")"
        if [ -n "$AUTO_STOP" ]; then
            echo "  Auto-stop: ${AUTO_STOP} (VM will stop automatically — use --no-auto-stop to disable)"
        fi
        echo ""
        echo "  Deploy (recommended):"
        echo "    ./llm.sh deploy ${VM_NAME}                                     # interactive wizard"
        echo "    ./llm.sh deploy ${VM_NAME} --backend llamacpp-docker --quant Q6_K --yes  # non-interactive"
        echo ""
        echo "  SSH:"
        echo "    gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT}"
        echo ""
        echo "  Manual upload (if not using llm.sh deploy):"
        echo "    gcloud compute scp --recurse vllm/ ${VM_NAME}:~ --zone=${ZONE} --project=${PROJECT}"
        echo "    gcloud compute scp --recurse docker/ ${VM_NAME}:~ --zone=${ZONE} --project=${PROJECT}"
        echo "    gcloud compute scp setup_llamacpp.sh ${VM_NAME}:~ --zone=${ZONE} --project=${PROJECT}"
        echo ""
        echo "  Stop when done (saves disk, stops billing):"
        echo "    ./llm.sh stop ${VM_NAME}"
        echo ""
        exit 0
    fi

    # VM creation failed — decide whether to retry or bail
    if echo "$OUTPUT" | grep -qi "PERMISSION_DENIED\|forbidden\|invalid.*argument\|invalid.*value\|not found.*service.account\|image.*not found"; then
        echo "  FATAL: $OUTPUT"
        echo ""
        echo "This error will occur in every zone. Fix the issue and retry."
        exit 1
    fi

    # Retryable: capacity exhausted, subnet missing in region, quota, etc.
    SHORT_ERR=$(echo "$OUTPUT" | grep -oi "ZONE_RESOURCE_POOL_EXHAUSTED\|stockout\|does not have enough resources\|quota.*exceeded\|subnet.*not found\|not exist in region" | head -1)
    if [ -n "$SHORT_ERR" ]; then
        echo "  ${SHORT_ERR} — skipping"
    else
        echo "  Failed: $(echo "$OUTPUT" | tail -1 | cut -c1-120)"
    fi
done

echo ""
echo "ERROR: Could not create VM in any of ${ZONE_COUNT} zones."
echo ""
echo "Possible causes:"
echo "  - All zones at capacity (try again later or try spot: --spot)"
echo "  - GPU quota exhausted (check with command below)"
echo "  - Subnet '${SUBNET}' doesn't exist in any region with ${GPU_TYPE}"
echo ""
echo "Check quotas:"
echo "  gcloud compute regions describe REGION --project=${PROJECT} \\"
echo "    --format='table(quotas.metric,quotas.limit,quotas.usage)' | grep -i gpu"
exit 1
