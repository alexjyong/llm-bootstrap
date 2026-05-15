a# Manual GCP VM Setup

For advanced users who want to create VMs manually instead of using `create_gpu_vm.sh`.

## Checking GPU Quota

```bash
# Check GPU quotas for a region
gcloud compute regions describe us-central1 --project=your-project-id \
    --format="table(quotas.metric,quotas.limit,quotas.usage)" \
    | grep -i gpu

# List which zones offer a specific GPU type
gcloud compute accelerator-types list --project=your-project-id \
    --filter="name:nvidia-tesla-a100" \
    --format="table(zone,name,description)"
```

GPU availability varies by zone and region. There is no CLI command for real-time capacity — you only find out a zone is exhausted when the create call fails with `ZONE_RESOURCE_POOL_EXHAUSTED`.

## Full VM Creation Command

From the GCP Console's "equivalent command line" output. **Zone, region, subnet, service account, disk policy, and image will vary** — use the Console UI to generate the correct command for your setup:

```bash
gcloud compute instances create my-llm \
    --project=your-project-id \
    --zone=us-central1-f \
    --machine-type=a2-highgpu-2g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=2,type=nvidia-tesla-a100 \
    --create-disk=auto-delete=yes,boot=yes,device-name=my-llm,disk-resource-policy=projects/your-project-id/regions/us-central1/resourcePolicies/default-schedule-1,image=projects/ml-images/global/images/common-cu129-ubuntu-2204-nvidia-580-v20260430,mode=rw,size=250,type=pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```

## Minimal Example

```bash
gcloud compute instances create my-llm \
    --project=your-project-id \
    --zone=us-east4-c \
    --machine-type=a2-highgpu-1g \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

## Raw gcloud Commands

```bash
gcloud compute instances list --project=your-project-id                              # List VMs
gcloud compute ssh VM_NAME --zone=ZONE --project=your-project-id                     # SSH
gcloud compute scp FILE VM_NAME:~ --zone=ZONE --project=your-project-id              # Upload file
gcloud compute scp --recurse docker/ VM_NAME:~ --zone=ZONE --project=your-project-id  # Upload folder
gcloud compute instances stop VM_NAME --zone=ZONE --project=your-project-id          # Stop
gcloud compute instances start VM_NAME --zone=ZONE --project=your-project-id         # Start
```
