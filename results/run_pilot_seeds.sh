#!/bin/bash
# waits for the XGBoost ablation to end, then runs the neural pilot sample one model at a time (8 GB Mac: never two jobs together)
cd "$(dirname "$0")/.."
while pgrep -f "src/gfs_ablation.py" >/dev/null; do sleep 20; done
for s in 1 2; do for m in c2 c1 convlstm; do
  .venv/bin/python -I -u src/train_nn.py $m 2022-01-01 2022-05-31 2023-01-01 2023-05-31 --max-train 100000 --max-val 100000 --epochs 4 --batch 512 --seed $s 2>&1 | grep -E "best val|params" | sed "s/^/seed $s /"
done; done
echo ALL DONE
