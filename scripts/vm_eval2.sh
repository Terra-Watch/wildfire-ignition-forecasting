#!/bin/bash
# Second evaluation pass: the paired comparisons only (all_runs, per_state, seeds are kept from the first pass). The run folders are already on this VM's disk.
exec >> /var/log/eval2.log 2>&1
set -x
shutdown -h +300
B=gs://kireeti-wildfire-507012/datasets_v4; R=$B/runs_v4_clean; OUT=$B/results_v4_clean
cd /home/user; rm -rf ml; mkdir -p ml res
gcloud storage cp $R/clean_abl/code.tgz /tmp/code.tgz && tar xzf /tmp/code.tgz -C ml
gcloud storage cp $OUT/all_runs.csv $OUT/per_state.csv $OUT/seeds.csv res/
for q in clean_cpu clean_rf clean_gpua clean_gpub clean_abl; do gcloud storage rsync -r --exclude='.*(model\.pt|model\.json|model\.joblib|code\.tgz|logs/.*)$' $R/$q runs_all/ && echo "$q synced"; done
cd ml
PAIRS="lstm_A:xgboost_A,mlp_A:xgboost_A,cnn_A:xgboost_A,convlstm_A:xgboost_A,c1_A:xgboost_A,c2_A:xgboost_A,rf_A:xgboost_A,lr_A:xgboost_A,c1_A:convlstm_A,c2_A:convlstm_A,xgboost_A:ref_fwi,xgboost_A:ref_climatology,xgboost_A:ref_persistence,xgboost_A:xgboost_only_fire_history,xgboost_A:xgboost_A_drop_fire_history,xgboost_A:xgboost_A_drop_ndvi,xgboost_A:xgboost_A_drop_weather,xgboost_A:xgboost_A_drop_landcover_terrain_people,xgboost_A:xgboost_A_drop_forecast,xgboost_A:xgboost_A_y_any,lstm_A:lstm_A_drop_fire_history,lstm_A:lstm_A_drop_ndvi,lstm_A:lstm_A_drop_weather,lstm_A:lstm_A_drop_landcover_terrain_people,lstm_A:lstm_A_drop_forecast"
/opt/v/bin/python src/evaluate.py --out /home/user/runs_all --res /home/user/res --splits test --paired-only --pairs "$PAIRS" && echo PAIRED_OK || echo PAIRED_FAILED
gcloud storage cp /home/user/res/paired.csv $OUT/ && echo "paired uploaded"
gcloud storage cp /var/log/eval2.log $OUT/logs/eval2.final.log
shutdown -h now
