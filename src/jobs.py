"""Job queues, one per machine role. Resumable: a run whose manifest says complete is skipped; a failed run is retried twice. Every run, every grid point and every restart stays on disk and in the
bucket and is listed by src/evaluate.py; nothing is deleted.

Hyperparameters: every grid point is a run with seed 0 on the training years; the setting with the best validation (2024) AUPRC is kept and that grid run is seed 0 of the final runs.
The grids have at most four settings for every model: LR four penalties, random forest two depths, XGBoost three depths, the six neural models two learning rates.
Roles (three projects): cpu (the 16-CPU machine: references, LR, XGBoost, ablations), rf (a second 16-CPU machine: random forest), gpua (MLP, LSTM, CNN, C2 and the neural ablations), gpub (ConvLSTM, C1).
Usage: python src/jobs.py cpu|rf|gpua|gpub [--out DIR] [--bucket gs://...] [--tiny] [--seeds 5]
"""
import argparse, json, os, shutil, subprocess, sys, time
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
A = None
LR_GRID = {f"C{c}": {"C": c} for c in (0.01, 0.1, 1, 10)}
XGB_GRID = {f"d{d}": {"max_depth": d} for d in (4, 6, 8)}
RF_GRID = {f"d{d}": {"max_depth": d} for d in (12, 20)}
NN_GRID = {"lr1e-3": {"lr": 1e-3}, "lr3e-4": {"lr": 3e-4}}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)
    with open(f"{A.out}/jobs.log", "a") as f: f.write(time.strftime("%FT%T ") + s + "\n")


def complete(d):
    try: return json.load(open(f"{d}/manifest.json")).get("complete", False)
    except Exception: return False


def upload(d, name):
    if not A.bucket: return
    tmp = f"{d}/../_up_{name}"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for f in os.listdir(d):
        if f.startswith("checkpoint"): continue
        (shutil.copytree if os.path.isdir(f"{d}/{f}") else shutil.copy)(f"{d}/{f}", f"{tmp}/{f}")
    r = subprocess.run(["gcloud", "storage", "cp", "-n", "-r", tmp, f"{A.bucket}/{name}"], capture_output=True, text=True); shutil.rmtree(tmp, ignore_errors=True)
    log(f"uploaded {name}" if r.returncode == 0 else f"UPLOAD FAILED {name}: {r.stderr[-300:]}")


def run(model, split, seed, hp, rid, out=None, extra=(), up=True):
    out = out or A.out; d = f"{out}/{rid}"
    if complete(d): log(f"skip {rid}"); return d
    cmd = [PY, f"{ROOT}/src/harness.py", "--model", model, "--split", split, "--seed", str(seed), "--run-id", rid, "--out", out, "--device", A.device, "--threads", str(A.threads), *extra]
    if hp: cmd += ["--hp", json.dumps(hp)]
    if A.tiny: cmd += ["--tiny"]
    for k in range(3):
        log("RUN " + " ".join(cmd[2:])); r = subprocess.run(cmd, cwd=ROOT)
        if complete(d): break
        log(f"run {rid} failed (exit {r.returncode}), attempt {k + 1}")
    else: raise SystemExit(f"run {rid} failed 3 times")
    if up: upload(d, rid)
    return d


def select(model, grid, seed0_id, extra=()):
    """Run every grid point (seed 0), keep the best validation AUPRC; that run is seed 0 of the final runs. Grid runs stay in _grid/ and are listed."""
    res = {}
    for name, hp in grid.items():
        d = run(model, "A", 0, hp, f"{seed0_id}__{name}", out=f"{A.out}/_grid", up=True, extra=extra); res[name] = json.load(open(f"{d}/manifest.json"))["best_val_auprc"]
    best = max(res, key=res.get); log(f"{seed0_id} grid validation AUPRC {res}; selected {best}")
    json.dump(dict(run=seed0_id, grid=res, selected=best, hp=grid[best]), open(f"{A.out}/hp_selected_{seed0_id}.json", "w"), indent=1)
    dst = f"{A.out}/{seed0_id}"
    if not complete(dst):
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(f"{A.out}/_grid/{seed0_id}__{best}", dst); m = json.load(open(f"{dst}/manifest.json")); m["run_id"] = seed0_id; json.dump(m, open(f"{dst}/manifest.json", "w"), indent=1)
    upload(dst, seed0_id)
    if A.bucket: subprocess.run(["gcloud", "storage", "cp", "-n", f"{A.out}/hp_selected_{seed0_id}.json", f"{A.bucket}/_hp/hp_selected_{seed0_id}.json"], capture_output=True)
    return grid[best]


def alias_c2025(model, seed, src_id, rid):
    """Rolling origin 2025 = train up to 2023, validate 2024, test 2025 = the main split: reuse the run's predictions under the C2025 names."""
    d = f"{A.out}/{rid}"
    if complete(d): return
    os.makedirs(f"{d}/predictions", exist_ok=True)
    for a, b in (("validation", "C2025_validation"), ("test", "C2025_test")):
        x = pd.read_parquet(f"{A.out}/{src_id}/predictions/{a}.parquet"); x["split"] = b; x.to_parquet(f"{d}/predictions/{b}.parquet", index=False)
    m = json.load(open(f"{A.out}/{src_id}/manifest.json")); m.update(run_id=rid, split="C2025", note=f"identical to {src_id} (same training window)"); json.dump(m, open(f"{d}/manifest.json", "w"), indent=1); upload(d, rid)


def refs():
    d = f"{A.out}/refs"
    if complete(d): log("skip refs"); return
    subprocess.run([PY, f"{ROOT}/src/run_refs.py", "--out", A.out], cwd=ROOT, check=True)
    for n in ("persistence", "climatology", "fwi"): upload(f"{A.out}/ref_{n}", f"ref_{n}")


def tabular(model, grid, tag, extra=()):
    """Single-cell and neighbour-feature variants of one tabular model: grid, five seeds, rolling-origin (LR and XGBoost only)."""
    out = {}
    for variant, ex in (("", ()),):      # neighbour-feature variants dropped by the owner on 2026-11-05 (DECISIONS.md amendment)
        ex = tuple(extra) + ex; hp = select(model, grid, f"{tag}{variant}_A_s0", extra=ex); out[variant] = hp
        for s in range(1, A.seeds if model != "lr" else 1): run(model, "A", s, hp, f"{tag}{variant}_A_s{s}", extra=ex)
        if model in ("lr", "xgboost") and not variant:                                   # rolling origin for the main (single-cell) models; the neighbour-feature set is reported separately
            for y in (2023, 2024):
                for s in range(A.seeds if model == "xgboost" else 1): run(model, f"C{y}", s, hp, f"{tag}{variant}_C{y}_s{s}", extra=ex)
            for s in range(A.seeds if model == "xgboost" else 1): alias_c2025(model, s, f"{tag}{variant}_A_s{s}", f"{tag}{variant}_C2025_s{s}")
    return out


def cpu():
    refs()
    lr = tabular("lr", LR_GRID, "lr")[""]
    xg = tabular("xgboost", XGB_GRID, "xgboost")[""]                                       # fire-history-only and the ablations use the main (single-cell) XGBoost setting
    run("lr", "A", 0, lr, "lr_only_fire_history", extra=("--only", "fire_history_only"))
    for s in range(A.seeds): run("xgboost", "A", s, xg, f"xgboost_only_fire_history_s{s}", extra=("--only", "fire_history_only"))
    for abl in ("fire_history", "ndvi", "weather", "landcover_terrain_people", "forecast"):
        for s in range(A.seeds): run("xgboost", "A", s, xg, f"xgboost_A_drop_{abl}_s{s}", extra=("--drop", abl))
    for s in range(A.seeds): run("xgboost", "A", s, xg, f"xgboost_A_y_any_s{s}", extra=("--label", "y_any"))
    log("cpu queue finished")


def local():
    """The part of the clean study that fits on a laptop (8 cores, 8 GB): logistic regression and XGBoost on the single-cell inputs, five seeds, plus the two ablations that test the cleaning (no fire history; the v3 label y_any). Ablation seeds: A.abl_seeds."""
    refs()
    lr = select("lr", LR_GRID, "lr_A_s0")
    xg = select("xgboost", XGB_GRID, "xgboost_A_s0")
    for s in range(1, A.seeds): run("xgboost", "A", s, xg, f"xgboost_A_s{s}")
    for s in range(A.abl_seeds): run("xgboost", "A", s, xg, f"xgboost_A_drop_fire_history_s{s}", extra=("--drop", "fire_history"))
    for s in range(A.abl_seeds): run("xgboost", "A", s, xg, f"xgboost_A_y_any_s{s}", extra=("--label", "y_any"))
    run("lr", "A", 0, lr, "lr_only_fire_history", extra=("--only", "fire_history_only"))
    log("local queue finished")


def rf():
    for variant, ex in (("", ()),):      # neighbour-feature variants dropped by the owner on 2026-11-05 (DECISIONS.md amendment)
        hp = select("rf", RF_GRID, f"rf{variant}_A_s0", extra=ex)
        for s in range(1, A.seeds): run("rf", "A", s, hp, f"rf{variant}_A_s{s}", extra=ex)
    log("rf queue finished")


def neural(models, ablate=False):
    for m in models:
        hp = select(m, NN_GRID, f"{m}_A_s0")
        for s in range(1, A.seeds): run(m, "A", s, hp, f"{m}_A_s{s}")
    log("neural models finished")


def gpua(): neural(("mlp", "lstm", "cnn", "c2")); log("gpua queue finished")
def gpub(): neural(("convlstm", "c1")); log("gpub queue finished")


def neural_ablation():
    """After every neural model has seed 0 on validation: the best one is run with each input group dropped, three seeds (run on gpua once gpub has finished)."""
    best = None
    for m in ("mlp", "lstm", "cnn", "convlstm", "c1", "c2"):
        p = fetch(f"{m}_A_s0/manifest.json", f"{m}_A_s0/manifest.json")
        if os.path.exists(p):
            v = json.load(open(p))["best_val_auprc"]
            if best is None or v > best[1]: best = (m, v)
    log(f"best neural model by validation AUPRC: {best}"); json.dump(dict(best=best), open(f"{A.out}/neural_best.json", "w"))
    m = best[0]; hp = json.load(open(fetch(f"hp_selected_{m}_A_s0.json", f"_hp/hp_selected_{m}_A_s0.json")))["hp"]
    for abl in ("fire_history", "ndvi", "weather", "landcover_terrain_people", "forecast"):
        for s in range(3): run(m, "A", s, hp, f"{m}_A_drop_{abl}_s{s}", extra=("--drop", abl))


def fetch(rel, bucket_rel):
    """A file written by another VM's queue: take it from the bucket when it is not on this disk."""
    f = f"{A.out}/{rel}"
    if not os.path.exists(f) and A.bucket: os.makedirs(os.path.dirname(f), exist_ok=True); subprocess.run(["gcloud", "storage", "cp", f"{A.bucket}/{bucket_rel}", f], capture_output=True)
    return f


DEV = {"ep30": {"epochs": 30}, "reg": {"drop": 0.1, "wd": 0.05}, "wide": {"hid": 64}}                 # one change each, one seed, validation only; the main runs keep their settings


def dev(model, extra_variants=()):
    """Development variants of a patch model (outside the pre-registered list, all reported). The base setting is the one selected on seed 0 in the main grid."""
    base = json.load(open(fetch(f"hp_selected_{model}_A_s0.json", f"_hp/hp_selected_{model}_A_s0.json")))["hp"]
    for name, hp in {**DEV, **dict(extra_variants)}.items(): run(model, "A", 0, {**base, **hp}, f"{model}_dev_{name}_A_s0")


def cell_path(model):
    """The patch model with an extra LSTM over the middle cell's own ten days (the cell-level model's path). Same treatment as every model: grid on seed 0, five seeds."""
    hp = select(model, {k: {**v, "cell": 1} for k, v in NN_GRID.items()}, f"{model}_cell_A_s0")
    for s in range(1, A.seeds): run(model, "A", s, hp, f"{model}_cell_A_s{s}")


def devb(): cell_path("convlstm"); dev("convlstm"); dev("c1")
def after(): cell_path("c2"); dev("c2", {"rounds3": {"rounds": 3}}); neural_ablation()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("role", choices=["cpu", "rf", "gpua", "gpub", "ablate", "after", "devb", "local"]); ap.add_argument("--out", default=f"{ROOT}/runs"); ap.add_argument("--bucket", default="")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--abl-seeds", dest="abl_seeds", type=int, default=3); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--device", default=""); ap.add_argument("--tiny", action="store_true")
    A = ap.parse_args(); A.device = A.device or ("cpu" if A.role in ("cpu", "rf", "local") else "cuda"); os.makedirs(A.out, exist_ok=True)
    {"cpu": cpu, "local": local, "rf": rf, "gpua": gpua, "gpub": gpub, "ablate": neural_ablation, "after": after, "devb": devb}[A.role]()
