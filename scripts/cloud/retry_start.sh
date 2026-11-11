#!/bin/bash
# Retry starting the stocked-out VMs in their own zone every 10 minutes (no other zone or project), for up to 3 hours.
for i in $(seq 18); do
  for x in "wf-cpu asia-south1-a tarakasrinivasande@gmail.com ace-sequencer-478304-h4" "wf-gpu-a asia-south1-b pattatejeswar@gmail.com jalari" "wf-gpu-b asia-south1-c dsvasi2005@gmail.com causal-bongo-508008-e5"; do
    set -- $x; s=$(gcloud compute instances describe $1 --zone=$2 --account=$3 --project=$4 --format="value(status)")
    [ "$s" = TERMINATED ] && gcloud compute instances start $1 --zone=$2 --account=$3 --project=$4 >/dev/null 2>&1 && echo "$(date +%T) started $1"
  done; sleep 600
done
