#!/bin/bash
# scripts/launch_ml_gpu.sh PROJECT RUN_ID ROLE [ZONE] -- one spot L4 VM (g2-standard-8) running the clean-study neural queue ROLE (gpua | gpub) from vm_job.sh. Resumable.
set -u
PRJ=$1; RID=$2; ROLE=$3; ZONES=${4:-"asia-south1-b asia-south1-c asia-south1-a"}; [ -z "$ZONES" ] && ZONES="asia-south1-b asia-south1-c asia-south1-a"; HOURS=${5:-12}; ML=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-ignition-forecasting; B=gs://kireeti-wildfire-507012/datasets_v4/runs_v4_clean
D=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md
ACC=$(grep -E "^\| $PRJ \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}')
[ -z "$ACC" ] && for A in $(gcloud auth list --format='value(account)' 2>/dev/null); do [ -n "$(gcloud projects list --account="$A" --filter="projectId=$PRJ" --format='value(projectId)' 2>/dev/null | head -1)" ] && ACC=$A; done
ACCB=$(sed -n '146p' $D | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
tar czf /tmp/ml_code_$RID.tgz -C $ML src scripts && gcloud storage cp /tmp/ml_code_$RID.tgz $B/$RID/code.tgz --account="$ACCB" --project=space-debris-507012 2>&1 | red | tail -1
for Z in $ZONES; do
  gcloud compute instances create mlg-${RID//_/-} --account="$ACC" --project=$PRJ --zone=$Z --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
    --provisioning-model=SPOT --instance-termination-action=STOP --max-run-duration=${HOURS}h --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB --boot-disk-type=pd-balanced --scopes=storage-rw --labels=purpose=wildfire-ml,owner=kireeti \
    --metadata=install-nvidia-driver=True,run_id=$RID,role=$ROLE,code=code.tgz,data_prefix=ml_ready_clean,runs_prefix=runs_v4_clean --metadata-from-file=startup-script=$ML/scripts/vm_job.sh 2>&1 | red | tail -2
  gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=mlg-${RID//_/-}" --format="value(name,zone.basename(),status)" 2>&1 | red | grep -q mlg-${RID//_/-} && { echo "ML_GPU_VM_CREATED $PRJ $RID $ROLE in $Z"; exit 0; }
done
echo "ML_GPU_VM_FAILED $PRJ $RID"; exit 1
