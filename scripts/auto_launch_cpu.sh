#!/bin/bash
# Waits for the clean cube in the research bucket and for a free CPU slot in ace-sequencer (quota 32 = two n2-standard-16), then starts the clean-study CPU queues one after the other: clean_cpu (cpu role: references, LR, XGBoost with seeds, rolling origins, ablations) and clean_rf (random forest). Markers in wildfire-ignition-forecasting/runs_clean_local/launched_* make it safe to re-run.
cd /Users/kireeti/Desktop/Projects/TerraWatch; D=wildfire-datasets-v3/DECISIONS.md; P=ace-sequencer-478304-h4; M=wildfire-ignition-forecasting/runs_clean_local
ACC=$(grep -E "^\| $P \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); ACCB=$(sed -n '146p' $D | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]+' | head -1); red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
while true; do
  [ -f $M/launched_cpu ] && [ -f $M/launched_rf ] && { echo "both CPU queues launched"; exit 0; }
  up=$(gcloud storage ls -l gs://kireeti-wildfire-507012/datasets_v4/ml_ready_clean/cube_patch.npz --account="$ACCB" 2>/dev/null | awk 'NR==1{print $1}')
  running=$(gcloud compute instances list --account="$ACC" --project=$P --filter="status=RUNNING" --format="value(name)" 2>/dev/null | grep -c .)
  if [ "$up" = "34224687" ] && [ "$running" -lt 2 ]; then
    if [ ! -f $M/launched_cpu ]; then wildfire-ignition-forecasting/scripts/launch_ml.sh $P clean_cpu cpu 2>&1 | red | tail -2 | tee /dev/stderr | grep -q ML_VM_CREATED && touch $M/launched_cpu && echo "launched clean_cpu"
    else wildfire-ignition-forecasting/scripts/launch_ml.sh $P clean_rf rf 2>&1 | red | tail -2 | tee /dev/stderr | grep -q ML_VM_CREATED && touch $M/launched_rf && echo "launched clean_rf"; fi
  fi
  sleep 300
done
