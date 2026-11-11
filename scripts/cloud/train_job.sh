#!/bin/bash
# Study queue, run as the VM's startup script (as root). What a VM runs comes from its instance metadata: run_id, role (cpu | rf | gpua | gpub | ablate).
# Inputs: code.tgz and the merged dataset in the wildfire bucket (read). Every finished run goes to gs://<bucket>/datasets_v4/runs_v4/<run_id>/<run dir> (no overwrite).
# Development phase only: trains on split_t == train, scores validation 2024 and test 2025 predictions to disk; the unseen states are scored later by scripts/05_score_blind_states.py.
# Restartable: finished runs are skipped. When the queue ends (or fails) the VM powers itself off; --max-run-duration is the backstop and the next start resumes.
set -uo pipefail
md() { curl -s -f -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
RUN_ID=$(md run_id); ROLE=$(md role); CODE=$(md code); [ -n "$RUN_ID" ] && [ -n "$ROLE" ] || { echo "no run_id or role"; exit 0; }
[ -f "/var/${RUN_ID}_${ROLE}.done" ] && { echo "role already done"; shutdown -h now; exit 0; }
B=gs://kireeti-wildfire-507012/datasets_v4; DS=$B/$(md data_prefix); [ "$DS" = "$B/" ] && DS=$B/ml_ready/dataset; RUNS=$B/$(md runs_prefix); [ "$RUNS" = "$B/" ] && RUNS=$B/runs_v4;   # metadata data_prefix / runs_prefix pick the clean study (ml_ready_clean/dataset_clean, runs_v4_clean)
 H=/home/user; LOG=/var/log/queue_${ROLE}.log
exec > >(tee -a "$LOG") 2>&1
finish() { sleep 3; gcloud storage cp -n "$LOG" "$RUNS/$RUN_ID/logs/$(hostname)_${ROLE}_$(date -u +%H%M).log"; shutdown -h now; }
trap finish EXIT
echo "queue start $(date -u +%FT%TZ) run=$RUN_ID role=$ROLE host=$(hostname)"; nvidia-smi -L || echo "no GPU on this machine"
mkdir -p $H/dataset $H/ml $H/runs; cd $H
{ gcloud storage cp "$RUNS/$RUN_ID/${CODE:-code.tgz}" /tmp/code.tgz && tar xzf /tmp/code.tgz -C ml || { echo "NO CODE"; exit 1; }; }
[ -f dataset/forest_all_gfs/cube.npy ] || gcloud storage cp -r $DS/forest_all_gfs $DS/scaler_forest_AG_train.npz dataset/ || { echo "DATA FETCH FAILED"; exit 1; }
if python3 -c "import torch" 2>/dev/null; then     # GPU image (pytorch-2-9-cu129-ubuntu-2204-nvidia-580): torch is already there
  python3 -m pip install -q pandas pyarrow scikit-learn xgboost joblib || { echo "PIP FAILED"; exit 1; }
else
  (command -v pip3 >/dev/null && python3 -m venv --help >/dev/null 2>&1) || { apt-get update -y && apt-get install -y python3-pip python3-venv; }
  python3 -m venv /opt/mlvenv && /opt/mlvenv/bin/pip install -q pandas pyarrow scikit-learn xgboost joblib $( [ "$ROLE" = cpu ] || [ "$ROLE" = rf ] || echo torch ) || { echo "PIP FAILED"; exit 1; }
  export PATH=/opt/mlvenv/bin:$PATH
fi
# clean study: the cube is the original one with 7 fire-history channels replaced (cube_patch.npz, 34 MB); the 3 GB clean cube itself never uploaded
if [ ! -f dataset/forest_all_gfs/.patched ] && gcloud storage ls $DS/cube_patch.npz >/dev/null 2>&1; then
  [ -f dataset/forest_all_gfs/cube.npy ] || gcloud storage cp $B/ml_ready/dataset/forest_all_gfs/cube.npy dataset/forest_all_gfs/cube.npy || { echo "CUBE FETCH FAILED"; exit 1; }
  gcloud storage cp $DS/cube_patch.npz /tmp/cube_patch.npz && python3 $H/ml/scripts/apply_cube_patch.py dataset/forest_all_gfs/cube.npy /tmp/cube_patch.npz && touch dataset/forest_all_gfs/.patched || { echo "PATCH FAILED"; exit 1; }
fi
export WILDFIRE_DATA=$H/dataset; cd ml
python3 -c "import xgboost; print('xgboost', xgboost.__version__)"
python3 scripts/02_train_models.py "$ROLE" --out $H/runs --bucket $RUNS/$RUN_ID --threads "$(nproc)" && touch "/var/${RUN_ID}_${ROLE}.done"
echo "queue end $(date -u +%FT%TZ)"
