#!/bin/bash
# scripts/launch_ml.sh PROJECT RUN_ID ROLE -- one CPU VM running the clean-study queue ROLE (cpu | rf) from vm_job.sh. Code and data are staged in the research bucket. Resumable: finished runs are skipped.
# The VM's service account needs write access to the research bucket (ace-sequencer, causal-bongo and jalari have it).
set -u
PRJ=$1; RID=$2; ROLE=$3; ML=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-ignition-forecasting; B=gs://kireeti-wildfire-507012/datasets_v4/runs_v4_clean
D=/Users/kireeti/Desktop/Projects/TerraWatch/wildfire-datasets-v3/DECISIONS.md
ACC=$(grep -E "^\| $PRJ \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); ACCB=$(sed -n '146p' $D | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1)
red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
tar czf /tmp/ml_code_$RID.tgz -C $ML src scripts && gcloud storage cp /tmp/ml_code_$RID.tgz $B/$RID/code.tgz --account="$ACCB" --project=space-debris-507012 2>&1 | red | tail -1
for Z in asia-south1-c asia-south1-a asia-south1-b; do
  gcloud compute instances create ml-${RID//_/-} --account="$ACC" --project=$PRJ --zone=$Z --machine-type=n2-standard-16 \
    --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced \
    --scopes=storage-rw --labels=purpose=wildfire-ml,owner=kireeti --max-run-duration=20h --instance-termination-action=STOP \
    --metadata=run_id=$RID,role=$ROLE,code=code.tgz,data_prefix=ml_ready_clean,runs_prefix=runs_v4_clean --metadata-from-file=startup-script=$ML/scripts/vm_job.sh 2>&1 | red | tail -2
  gcloud compute instances list --account="$ACC" --project=$PRJ --filter="name=ml-${RID//_/-}" --format="value(name,zone.basename(),status)" 2>&1 | red | grep -q ml-${RID//_/-} && { echo "ML_VM_CREATED $PRJ $RID $ROLE in $Z"; exit 0; }
done
echo "ML_VM_FAILED $PRJ $RID"; exit 1
