#!/bin/bash
# CPU VM: freeze record (one SHA-256 over the settings of every finished run), then the unseen states scored ONCE (WILDFIRE_FINAL=1), then the evaluation tables for the blind split. Uploads freeze.json and the blind tables.
exec >> /var/log/blind.log 2>&1
set -x
shutdown -h +600
B=gs://kireeti-wildfire-507012/datasets_v4; R=$B/runs_v4_clean; DS=$B/ml_ready_clean; OUT=$B/results_v4_clean
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
python3 -m venv /opt/v && /opt/v/bin/pip install -q pandas pyarrow scikit-learn numpy scipy joblib xgboost torch --extra-index-url https://download.pytorch.org/whl/cpu
mkdir -p /home/user/ml /home/user/runs_all /home/user/res_blind /home/user/dataset; cd /home/user
gcloud storage cp $R/clean_abl/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C ml
( while true; do sleep 300; gcloud storage cp /var/log/blind.log $OUT/logs/blind.log >/dev/null 2>&1; done ) &
gcloud storage cp -r $DS/forest_unseen_gfs $DS/scaler_forest_AG_train.npz dataset/ && echo "unseen data fetched"
gcloud storage cp -r $DS/forest_all_gfs dataset/ 2>/dev/null; ls dataset
for q in clean_cpu clean_rf clean_gpua clean_gpub clean_abl; do
  gcloud storage rsync -r --exclude='.*(code\.tgz|logs/.*|_grid/.*)$' $R/$q runs_all/ && echo "$q synced" || echo "$q SYNC FAILED"
done
export WILDFIRE_DATA=/home/user/dataset; cd ml
/opt/v/bin/python scripts/04_freeze_protocol.py --out /home/user/runs_all --res /home/user/res_blind && cat /home/user/res_blind/freeze.json | head -3
gcloud storage cp /home/user/res_blind/freeze.json $OUT/freeze.json
WILDFIRE_FINAL=1 /opt/v/bin/python scripts/05_score_blind_states.py --out /home/user/runs_all && echo BLIND_SCORED || echo BLIND_FAILED
PAIRS="lstm_A:xgboost_A,mlp_A:xgboost_A,cnn_A:xgboost_A,convlstm_A:xgboost_A,c1_A:xgboost_A,c2_A:xgboost_A,rf_A:xgboost_A,lr_A:xgboost_A,xgboost_A:ref_fwi,xgboost_A:ref_climatology,xgboost_A:ref_persistence,xgboost_A:xgboost_only_fire_history"
/opt/v/bin/python scripts/03_evaluate_runs.py --out /home/user/runs_all --res /home/user/res_blind --splits blind --pairs "$PAIRS" && echo BLIND_EVAL_OK || echo BLIND_EVAL_FAILED
mkdir -p /home/user/res_blind/up; for f in all_runs.csv per_state.csv seeds.csv paired.csv freeze.json; do [ -f /home/user/res_blind/$f ] && cp /home/user/res_blind/$f /home/user/res_blind/up/blind_$f; done
gcloud storage cp /home/user/res_blind/up/* $OUT/ && echo "blind tables uploaded"
gcloud storage cp /var/log/blind.log $OUT/logs/blind.final.log
shutdown -h now
