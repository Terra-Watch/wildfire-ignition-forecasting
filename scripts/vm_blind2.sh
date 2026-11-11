#!/bin/bash
# CPU VM, second blind pass: the same frozen models scored again (deterministic; checked against the first tables) so that the blind predictions are kept; then calibration columns, intervals. Original text: freeze record (one SHA-256 over the settings of every finished run), then the unseen states scored ONCE (WILDFIRE_FINAL=1), then the evaluation tables for the blind split. Uploads freeze.json and the blind tables.
exec >> /var/log/blind2.log 2>&1
set -x
shutdown -h +600
B=gs://kireeti-wildfire-507012/datasets_v4; R=$B/runs_v4_clean; DS=$B/ml_ready_clean; OUT=$B/results_v4_clean
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
python3 -m venv /opt/v && /opt/v/bin/pip install -q pandas pyarrow scikit-learn numpy scipy joblib xgboost torch --extra-index-url https://download.pytorch.org/whl/cpu
mkdir -p /home/user/ml /home/user/runs_all /home/user/res2 /home/user/dataset; cd /home/user
gcloud storage cp $R/clean_abl/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C ml
( while true; do sleep 300; gcloud storage cp /var/log/blind2.log $OUT/logs/blind2.log >/dev/null 2>&1; done ) &
gcloud storage cp -r $DS/forest_unseen_gfs $DS/scaler_forest_AG_train.npz dataset/ && echo "unseen data fetched"
gcloud storage cp -r $DS/forest_all_gfs dataset/ 2>/dev/null; ls dataset
for q in clean_cpu clean_rf clean_gpua clean_gpub clean_abl; do
  gcloud storage rsync -r --exclude='.*(code\.tgz|logs/.*|_grid/.*)$' $R/$q runs_all/ && echo "$q synced" || echo "$q SYNC FAILED"
done
export WILDFIRE_DATA=/home/user/dataset; cd ml
/opt/v/bin/python src/freeze.py --out /home/user/runs_all --res /home/user/res2 && python3 -c "import json;print('FREEZE_NOW',json.load(open('/home/user/res2/freeze.json'))['sha256'])"
WILDFIRE_FINAL=1 /opt/v/bin/python src/score_unseen.py --out /home/user/runs_all && echo BLIND_SCORED || echo BLIND_FAILED
PAIRS="lstm_A:xgboost_A,mlp_A:xgboost_A,cnn_A:xgboost_A,convlstm_A:xgboost_A,c1_A:xgboost_A,c2_A:xgboost_A,rf_A:xgboost_A,lr_A:xgboost_A,xgboost_A:ref_fwi,xgboost_A:ref_climatology,xgboost_A:ref_persistence,xgboost_A:xgboost_only_fire_history"
/opt/v/bin/python src/evaluate.py --out /home/user/runs_all --res /home/user/res2 --splits validation,blind --pairs "$PAIRS" && echo BLIND_EVAL_OK || echo BLIND_EVAL_FAILED
/opt/v/bin/python src/intervals.py --runs /home/user/runs_all --out /home/user/res2/intervals.csv --splits test,blind --groups xgboost_A,lstm_A,rf_A,c2_A,c1_A,convlstm_A,mlp_A,cnn_A,lr_A,ref_climatology,ref_persistence,ref_fwi && echo INTERVALS_OK || echo INTERVALS_FAILED
gcloud storage cp $OUT/blind_seeds.csv /tmp/first_blind_seeds.csv
/opt/v/bin/python - <<'PY'
import pandas as pd
a=pd.read_csv("/tmp/first_blind_seeds.csv"); b=pd.read_csv("/home/user/res2/seeds.csv")
k=["group","model","split","stratum"]; a=a[(a.split=="blind")].set_index(k); b=b[(b.split=="blind")].set_index(k)
j=a[["auprc_mean"]].join(b[["auprc_mean"]],rsuffix="_2",how="inner"); print("REPRODUCED groups",len(j),"max abs diff",float((j.auprc_mean-j.auprc_mean_2).abs().max()))
PY
mkdir -p /home/user/res2/up; for f in all_runs.csv per_state.csv seeds.csv paired.csv intervals.csv; do [ -f /home/user/res2/$f ] && cp /home/user/res2/$f /home/user/res2/up/blind2_$f; done
gcloud storage cp /home/user/res2/up/* $OUT/ && echo "tables uploaded"
for d in /home/user/runs_all/*/; do n=$(basename $d); [ -f $d/predictions/blind.parquet ] && gcloud storage cp $d/predictions/blind.parquet $OUT/blind_predictions/$n.parquet >/dev/null 2>&1; done; echo "blind predictions kept"
gcloud storage cp /var/log/blind2.log $OUT/logs/blind2.final.log
shutdown -h now
