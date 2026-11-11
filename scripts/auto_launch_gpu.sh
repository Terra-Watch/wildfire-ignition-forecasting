#!/bin/bash
# Neural queues of the clean study on the L4 of ace-sequencer (user, 2026-11-05: "Utilize the L4 from ace seq"; credit $300). One L4 per project, so gpua (mlp, lstm, cnn, c2) runs first and gpub (convlstm, c1) after it ends. ace has a global CPU quota of 32: a g2-standard-8 needs 8, so it starts when at most one n2-standard-16 is running there. Markers runs_clean_local/launched_gpua / launched_gpub; safe to re-run. Caps: 20 h each (spot).
cd /Users/kireeti/Desktop/Projects/TerraWatch; D=wildfire-datasets-v3/DECISIONS.md; M=wildfire-ignition-forecasting/runs_clean_local; P=ace-sequencer-478304-h4
ACC=$(grep -E "^\| $P \| " $D | head -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); red() { sed -E 's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+/<email>/g'; }
running() { gcloud compute instances list --account="$ACC" --project=$P --filter="status=RUNNING OR status=STAGING OR status=PROVISIONING" --format="value(name)" 2>/dev/null | grep -c .; }
status() { gcloud compute instances list --account="$ACC" --project=$P --filter="name=$1" --format="value(status)" 2>/dev/null; }
while true; do
  [ -f $M/launched_gpua ] && [ -f $M/launched_gpub ] && { echo "both GPU queues launched"; exit 0; }
  n=$(running)
  if [ ! -f $M/launched_gpua ] && [ "$n" -le 1 ]; then wildfire-ignition-forecasting/scripts/launch_ml_gpu.sh $P clean_gpua gpua "" 20 2>&1 | red | tail -2 | tee /dev/stderr | grep -q ML_GPU_VM_CREATED && touch $M/launched_gpua && echo "launched clean_gpua in ace-sequencer"
  elif [ -f $M/launched_gpua ] && [ ! -f $M/launched_gpub ] && [ "$(status mlg-clean-gpua)" = "TERMINATED" ] && [ "$n" -le 1 ]; then wildfire-ignition-forecasting/scripts/launch_ml_gpu.sh $P clean_gpub gpub "" 20 2>&1 | red | tail -2 | tee /dev/stderr | grep -q ML_GPU_VM_CREATED && touch $M/launched_gpub && echo "launched clean_gpub in ace-sequencer"; fi
  sleep 300
done
