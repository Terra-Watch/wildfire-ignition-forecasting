#!/bin/bash
# CPU VM: evaluation of every forecast run (validation and test; the blind states are scored later, once, after the freeze). Gathers the run folders of all queues (without model files), runs scripts/03_evaluate_runs.py with the pre-registered pairs, uploads the tables.
exec >> /var/log/eval.log 2>&1
set -x
shutdown -h +600
B=gs://kireeti-wildfire-507012/datasets_v4; R=$B/runs_v4_clean; OUT=$B/results_v4_clean
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
python3 -m venv /opt/v && /opt/v/bin/pip install -q pandas pyarrow scikit-learn numpy scipy joblib
mkdir -p /home/user/ml /home/user/runs_all /home/user/res; cd /home/user
gcloud storage cp $R/clean_abl/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C ml
( while true; do sleep 300; gcloud storage cp /var/log/eval.log $OUT/logs/eval.log >/dev/null 2>&1; done ) &
for q in clean_cpu clean_rf clean_gpua clean_gpub clean_abl; do
  gcloud storage rsync -r --exclude='.*(model\.pt|model\.json|model\.joblib|code\.tgz|logs/.*)$' $R/$q runs_all/ && echo "$q synced" || echo "$q SYNC FAILED"
done
ls runs_all | wc -l; du -sh runs_all
cd ml
PAIRS="lstm_A:xgboost_A,mlp_A:xgboost_A,cnn_A:xgboost_A,convlstm_A:xgboost_A,c1_A:xgboost_A,c2_A:xgboost_A,rf_A:xgboost_A,lr_A:xgboost_A,c1_A:convlstm_A,c2_A:convlstm_A,xgboost_A:ref_fwi,xgboost_A:ref_climatology,xgboost_A:ref_persistence,xgboost_A:xgboost_only_fire_history,xgboost_A:xgboost_A_drop_fire_history,xgboost_A:xgboost_A_drop_ndvi,xgboost_A:xgboost_A_drop_weather,xgboost_A:xgboost_A_drop_landcover_terrain_people,xgboost_A:xgboost_A_drop_forecast,xgboost_A:xgboost_A_y_any,lstm_A:lstm_A_drop_fire_history,lstm_A:lstm_A_drop_ndvi,lstm_A:lstm_A_drop_weather,lstm_A:lstm_A_drop_landcover_terrain_people,lstm_A:lstm_A_drop_forecast"
/opt/v/bin/python scripts/03_evaluate_runs.py --out /home/user/runs_all --res /home/user/res --splits validation,test --pairs "$PAIRS" && echo EVAL_OK || echo EVAL_FAILED
gcloud storage cp -r /home/user/res/* $OUT/ && echo "results uploaded"
gcloud storage cp /var/log/eval.log $OUT/logs/eval.final.log
shutdown -h now
