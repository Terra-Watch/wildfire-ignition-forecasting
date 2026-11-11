#!/bin/bash
# CPU VM: the remaining analyses of the forecast paper: permutation importance (XGBoost, validation year), test and blind AUPRC by land-cover class of the cell, and the risk map of one test day.
exec >> /var/log/extras.log 2>&1
set -x
shutdown -h +420
B=gs://kireeti-wildfire-507012/datasets_v4; R=$B/runs_v4_clean; DS=$B/ml_ready_clean; OUT=$B/results_v4_clean/extras
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-pip python3-venv
python3 -m venv /opt/v && /opt/v/bin/pip install -q pandas pyarrow scikit-learn numpy scipy joblib xgboost matplotlib
H=/home/user; W=$H/wf; mkdir -p $W/wildfire-ignition-forecasting $W/terrawatch-datasets/dataset $H/dataset $H/runs $H/res/figures; cd $H
gcloud storage cp $R/clean_abl/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C $W/wildfire-ignition-forecasting
( while true; do sleep 300; gcloud storage cp /var/log/extras.log $OUT/logs/extras.log >/dev/null 2>&1; done ) &
gcloud storage cp -r $DS/forest_all_gfs $DS/scaler_forest_AG_train.npz dataset/ || echo "DATA FETCH FAILED"
# the clean study's cube: original cube with 7 fire-history channels replaced (cube_patch.npz)
gcloud storage cp $B/ml_ready/dataset/forest_all_gfs/cube.npy dataset/forest_all_gfs/cube.npy && gcloud storage cp $DS/cube_patch.npz /tmp/cube_patch.npz && /opt/v/bin/python $W/wildfire-ignition-forecasting/scripts/apply_cube_patch.py dataset/forest_all_gfs/cube.npy /tmp/cube_patch.npz && echo CUBE_PATCHED || echo CUBE_PATCH_FAILED
ln -sfn $H/dataset/forest_all_gfs $W/terrawatch-datasets/dataset/forest_all_gfs
for q in clean_cpu clean_rf clean_gpua clean_gpub; do gcloud storage rsync -r --exclude='.*(model\.pt|code\.tgz|logs/.*|_grid/.*)$' $R/$q runs/ && echo "$q synced"; done
export WILDFIRE_DATA=$H/dataset; cd $W/wildfire-ignition-forecasting
/opt/v/bin/python scripts/08_permutation_importance.py --model xgboost --path $H/runs/xgboost_A_s0/model.json --out $H/res/permutation_xgboost.csv && echo PERM_OK || echo PERM_FAILED
/opt/v/bin/python scripts/09_landcover_split.py --runs $H/runs --out $H/res/landcover_split_test.csv --split test && echo LC_TEST_OK || echo LC_TEST_FAILED
WILDFIRE_FINAL=1 /opt/v/bin/python scripts/09_landcover_split.py --runs $H/runs --out $H/res/landcover_split_blind.csv --split blind && echo LC_BLIND_OK || echo LC_BLIND_FAILED
/opt/v/bin/python scripts/10_risk_map.py --runs $H/runs --out $H/res/figures && echo MAP_OK || echo MAP_FAILED
gcloud storage cp -r $H/res/* $OUT/ && echo "extras uploaded"
gcloud storage cp /var/log/extras.log $OUT/logs/extras.final.log
shutdown -h now
