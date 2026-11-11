"""One entry point that trains one model on forest_all_gfs and writes its results. Resumable: a run whose manifest says complete is skipped.

  python scripts/train_one_model.py --model xgboost --split A --seed 0 --hp '{"max_depth": 6}' --run-id xgboost_A_s0 [--drop forecast] [--label y_any] [--device cuda]

Rows: the valid_primary rows of the 17 training states (valid_alt rows and y_any when --label y_any). Splits by label date through split_t:
  A      train = split_t train (labels up to 2023-12-31), validation = 2024, test = 2025
  C<Y>   rolling origin: train = labels up to (Y-2)-12-31, validation = Y-1, test = Y   (LR and XGBoost only)
The forest_unseen states are never opened here (scripts/05_score_blind_states.py does that, after every model is frozen).
Features: all channels of the final input set, scaled with scaler_forest_AG_train.npz (training rows only); non-finite values become 0 and the image-like inputs carry a flag for "neighbour exists".
Imbalance: positive rows get the weight w = sqrt(n_neg / n_pos) of the training rows, in the loss only (no sampling).
A run writes <out>/<run_id>/{manifest.json, config.json, history.json, train.log, predictions/<split>.parquet, model.*}.
"""
import argparse, hashlib, json, os, platform, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("WILDFIRE_DATA") or os.path.join(os.path.dirname(ROOT), "wildfire-dataset", "dataset")
DATASET = "forest_all_gfs"
FIRE = ["fire_days_7d", "fire_days_30d", "days_since_last_fire", "has_history", "fire_days_7d_f", "fire_days_30d_f", "days_since_last_fire_f"]
GROUPS = {
    "fire_history": FIRE, "ndvi": ["ndvi", "ndvi_age_days"], "roads": ["distance_to_roads_km"], "human": ["population_density", "distance_to_roads_km"],
    "weather": ["t2m_max", "t2m_mean", "rh_min_proxy", "vpd_max_proxy", "wind_speed_meanvec", "precip_day", "precip_7d", "precip_14d", "precip_30d", "days_since_rain", "soil_moisture"],
    "landcover_terrain_people": ["forest_fraction", "shrub_fraction", "grass_fraction", "crop_fraction", "elevation", "slope", "population_density", "distance_to_roads_km"],
    "forecast": ["gfs_t2m_max", "gfs_t2m_mean", "gfs_rh_min", "gfs_vpd_max", "gfs_wind_mean", "gfs_gust_max", "gfs_rain_mm", "gfs_soil_0_10"]}
ONLY = {"fire_history_only": FIRE + ["doy_sin", "doy_cos"]}                       # the fire-history-only reference models
DEFAULT_HP = {"lr": {"C": 1.0}, "rf": {"max_depth": 12}, "xgboost": {"max_depth": 6}, "mlp": {"lr": 1e-3}, "lstm": {"lr": 1e-3}, "cnn": {"lr": 1e-3}, "convlstm": {"lr": 1e-3}, "c1": {"lr": 1e-3}, "c2": {"lr": 1e-3}}
RING = [6, 7, 8, 11, 13, 16, 17, 18]                                               # the 8 cells around the middle of the 5 x 5 patch


def log_to(path):
    f = open(path, "a")
    def log(s): print(s, flush=True); f.write(s + "\n"); f.flush()
    return log


def split_dates(split):
    if split == "A": return None
    y = int(split[1:]); return dict(train=(None, f"{y - 2}-12-31"), validation=(f"{y - 1}-01-01", f"{y - 1}-12-31"), test=(f"{y}-01-01", f"{y}-12-31"))


class Cube:
    """The scaled cube of one dataset on a device (days [t0, t1] only), with an `ok` mask and the 5 x 5 neighbour table. `keep` selects channels."""
    def __init__(self, name, keep_names, device, t_range=None, scaler=None, use_torch=True):
        self.use_torch, self.device, self.name = use_torch, device, name; d = f"{DATA}/{name}"
        if use_torch: import torch; self.torch = torch                                 # tabular models run on numpy only (torch and XGBoost crash together on a Mac)
        self.rows = pd.read_parquet(f"{d}/rows.parquet"); self.cells = pd.read_parquet(f"{d}/cells.parquet"); raw = np.load(f"{d}/cube.npy", mmap_mode="r")
        self.scaler_path = scaler or f"{DATA}/scaler_forest_AG_train.npz"; z = np.load(self.scaler_path, allow_pickle=True); mean, sd = z["mean"].astype(np.float64), z["sd"].astype(np.float64); self.names = [str(n) for n in z["names"]]
        assert len(self.names) == raw.shape[2], "scaler and cube differ in channel count"
        self.keep_idx = [self.names.index(n) for n in keep_names]; self.keep_names = list(keep_names); self.mean, self.sd = mean, sd
        D = raw.shape[0]; self.t0, self.t1 = (0, D - 1) if t_range is None else (max(0, t_range[0]), min(D - 1, t_range[1])); n = self.t1 - self.t0 + 1; C = raw.shape[1]
        X = np.empty((n, C, len(self.keep_idx)), dtype=np.float32); ok = np.empty((n, C), dtype=bool)
        for a in range(0, n, 128):
            x = (np.asarray(raw[self.t0 + a:self.t0 + min(a + 128, n)], dtype=np.float64) - mean) / sd; f = np.isfinite(x).all(-1); x[~np.isfinite(x)] = 0
            X[a:a + len(x)] = x[..., self.keep_idx].astype(np.float32); ok[a:a + len(x)] = f
        r, c = self.cells.r.values, self.cells.c.values; g = np.full((r.max() + 5, c.max() + 5), -1, dtype=np.int64); g[r + 2, c + 2] = np.arange(len(r)); off = np.arange(-2, 3)
        nbr = g[(r[:, None, None] + 2 + off[None, :, None]), (c[:, None, None] + 2 + off[None, None, :])].reshape(len(r), 25)
        if use_torch: self.X, self.ok, self.nbr = (self.torch.from_numpy(v).to(device) for v in (X, ok, nbr))
        else: self.X, self.ok, self.nbr = X, ok, nbr
        full = {nm: i for i, nm in enumerate(self.names)}; self.pos = {nm: self.keep_names.index(nm) for nm in FIRE if nm in self.keep_names}
        self.f7f_raw = None; self.nbfeat = False                                      # nbfeat: the MLP and LSTM get the neighbour fire-history features too
        if "fire_days_7d_f" in full:                                                  # raw value of fire_days_7d_f for the strict new-ignition stratum, read straight from the cube
            self.f7f = np.asarray(raw[self.t0:self.t1 + 1, :, full["fire_days_7d_f"]], dtype=np.float32)

    def gather(self, view, t, cp):
        """t, cp: int64 tensors [B] (t is a cube day). tabular [B,F]; sequence [B,10,F]; patch [B,F+1,5,5]; patch_sequence [B,10,F+1,5,5]"""
        t0 = t; t = t - self.t0
        if view == "tabular": return torch_cat([self.X[t, cp], self.neighbour_features(t0, cp)]) if self.nbfeat else self.X[t, cp]
        torch = self.torch
        if view == "sequence":
            x = self.X[t[:, None] - torch.arange(9, -1, -1, device=t.device)[None, :], cp[:, None]]
            if self.nbfeat: x = torch.cat([x, torch.stack([self.neighbour_features(t0 - k, cp) for k in range(9, -1, -1)], 1)], -1)   # neighbour features of each of the ten days
            return x
        nb = self.nbr[cp]; present = nb >= 0; nbc = nb.clamp(min=0)
        if view == "patch":
            x = self.X[t[:, None], nbc]; v = present & self.ok[t[:, None], nbc]; x = x * v[..., None]
            return torch.cat([x, v[..., None].to(x.dtype)], -1).reshape(len(t), 5, 5, -1).permute(0, 3, 1, 2).contiguous()
        tt = t[:, None] - torch.arange(9, -1, -1, device=t.device)[None, :]
        x = self.X[tt[:, :, None], nbc[:, None, :]]; v = present[:, None, :] & self.ok[tt[:, :, None], nbc[:, None, :]]; x = x * v[..., None]
        return torch.cat([x, v[..., None].to(x.dtype)], -1).reshape(len(t), 10, 5, 5, -1).permute(0, 1, 4, 2, 3).contiguous()

    def neighbour_features(self, t, cp):
        """Mean of the fire-history channels over the 8 cells around (those that exist; the cell's own value when none) and the count of those with a forest detection in the last 7 days."""
        t = t - self.t0; ix = [self.pos[n] for n in FIRE if n in self.pos]; nb = self.nbr[cp][:, RING]; nbc = nb.clip(0) if not self.use_torch else nb.clamp(min=0)
        present = (nb >= 0) & self.ok[t[:, None], nbc]; f = self.X[t[:, None], nbc][:, :, ix]; own = self.X[t, cp][:, ix]
        j = self.pos["fire_days_7d_f"]; k = self.names.index("fire_days_7d_f"); raw = self.X[t[:, None], nbc][:, :, j] * float(self.sd[k]) + float(self.mean[k])
        if not self.use_torch:
            m = present[..., None].astype(np.float32); n_ = m.sum(1); mean = np.where(n_ > 0, (f * m).sum(1) / np.maximum(n_, 1), own)
            return np.concatenate([mean, ((raw > 0.5) & present).sum(1, keepdims=True).astype(np.float32)], 1)
        torch = self.torch; m = present.to(f.dtype)[..., None]; mean = torch.where(m.sum(1) > 0, (f * m).sum(1) / m.sum(1).clamp(min=1), own)
        return torch.cat([mean, ((raw > 0.5) & present).sum(1, keepdim=True).to(f.dtype)], 1)


torch_cat = lambda xs: __import__('torch').cat(xs, 1)


def pick_rows(cube, spec_dates, split, label):
    """Row table of one split (training states only)."""
    r = cube.rows; m = (r.valid_alt if label == "y_any" else r.valid_primary).values & (r.split_t != "none").values
    if spec_dates is None: m &= (r.split_t == split).values
    else:
        lo, hi = spec_dates[split]; ld = pd.to_datetime(r.label_date).values; m &= ld <= np.datetime64(hi)
        if lo: m &= ld >= np.datetime64(lo)
    out = r[m].copy(); out["y"] = (out.y_any if label == "y_any" else out.y_forest).astype(np.int8); return out.reset_index(drop=True)


def tab_matrix(cube, rs, neighbours):
    out = []
    for a in range(0, len(rs), 1_000_000):
        s = rs.iloc[a:a + 1_000_000]; t = s.t.values.astype(np.int64); cp = s.cell_idx.values.astype(np.int64); x = cube.gather("tabular", t, cp)
        if neighbours: x = np.concatenate([x, cube.neighbour_features(t, cp)], 1)
        out.append(x.astype(np.float32))
    return np.concatenate(out)


def predict_net(net, cube, view, rs, batch, device, amp):
    import torch
    net.eval(); out = np.empty(len(rs), np.float32); t = torch.from_numpy(rs.t.values.astype(np.int64)).to(device); cp = torch.from_numpy(rs.cell_idx.values.astype(np.int64)).to(device)
    with torch.no_grad():
        for a in range(0, len(rs), batch):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp): z = net(cube.gather(view, t[a:a + batch], cp[a:a + batch]))
            out[a:a + batch] = torch.sigmoid(z.float()).cpu().numpy()
    return out


def fit_neural(a, cube, tr, va, w, run_dir, log):
    import torch
    from sklearn.metrics import average_precision_score
    from wildfire_forecast import neural_nets as nets
    cfg = dict(nets.NEURAL[a.model]); hp = json.loads(a.hp) if a.hp else dict(DEFAULT_HP[a.model]); cfg.update(hp); cfg.setdefault("lr", 1e-3); dev = torch.device(a.device); amp = dev.type == "cuda" and not a.no_amp
    cube.nbfeat = bool(a.neighbours and a.model in ("mlp", "lstm") and "fire_days_7d_f" in cube.pos); torch.manual_seed(a.seed); np.random.seed(a.seed); F = cube.X.shape[-1] + (len(cube.pos) + 1 if cube.nbfeat else 0); kw = {k: cfg[k] for k in ("hid", "drop", "rounds", "cell") if k in cfg}; net = nets.build(a.model, F, **kw).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=dev)); view = cfg["view"]; B = cfg["batch"]
    t = torch.from_numpy(tr.t.values.astype(np.int64)).to(dev); cp = torch.from_numpy(tr.cell_idx.values.astype(np.int64)).to(dev); y = torch.from_numpy(tr.y.values.astype(np.float32)).to(dev)
    ck = f"{run_dir}/checkpoint.pt"; hist = []; best = -1.0; bad = 0; ep0 = 0; best_state = None
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev); net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); hist, best, bad, ep0, best_state = s["hist"], s["best"], s["bad"], s["epoch"], s["best_state"]; log(f"resumed at epoch {ep0}")
    n_rows = a.epoch_rows or len(tr)
    for ep in range(ep0, cfg["epochs"]):
        if bad >= 3: break
        g = torch.Generator(device="cpu"); g.manual_seed(a.seed * 1000 + ep); perm = torch.randperm(len(tr), generator=g)[:n_rows].to(dev); net.train(); tot = 0.0; nb = 0; te = time.time()
        for i in range(0, len(perm), B):
            ix = perm[i:i + B]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp): z = net(cube.gather(view, t[ix], cp[ix]))
            loss = lossf(z.float(), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss); nb += 1
            if a.max_steps and nb >= a.max_steps: break
        pv = predict_net(net, cube, view, va, 8192, dev, amp); ap = float(average_precision_score(va.y.values, pv))
        hist.append(dict(epoch=ep + 1, train_loss=tot / nb, val_auprc=ap, seconds=round(time.time() - te), steps=nb)); log(str(hist[-1]))
        if ap > best: best, bad, best_state = ap, 0, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else: bad += 1
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), hist=hist, best=best, bad=bad, epoch=ep + 1, best_state=best_state), ck); json.dump(hist, open(f"{run_dir}/history.json", "w"), indent=1)
    net.load_state_dict(best_state); torch.save(dict(state=best_state, model=a.model, F=F, cfg=cfg, channels=cube.keep_names, nbfeat=cube.nbfeat), f"{run_dir}/model.pt")
    if os.path.exists(ck): os.remove(ck)
    return (lambda rs, cb=cube: predict_net(net, cb, view, rs, 8192, dev, amp)), dict(n_params=int(sum(p.numel() for p in net.parameters())), epochs_or_trees=len(hist), best_val_auprc=best, cfg=cfg, epoch_rows=n_rows)


def fit_tabular(a, cube, tr, va, w, run_dir, log):
    from sklearn.metrics import average_precision_score
    nb = a.neighbours and "fire_days_7d_f" in cube.pos; Xtr, ytr = tab_matrix(cube, tr, nb), tr.y.values; hp = json.loads(a.hp) if a.hp else dict(DEFAULT_HP[a.model]); nthr = a.threads
    if a.model == "lr":
        import joblib
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=hp["C"], max_iter=hp.get("max_iter", 2000), class_weight={0: 1.0, 1: w}); m.fit(Xtr, ytr); joblib.dump(m, f"{run_dir}/model.joblib")
        pr = lambda rs, cb=cube: m.predict_proba(tab_matrix(cb, rs, nb))[:, 1]; info = dict(n_params=int(m.coef_.size + 1), epochs_or_trees=int(m.n_iter_[0]), cfg=hp, converged=bool(m.n_iter_[0] < hp.get("max_iter", 2000)))
    elif a.model == "rf":
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=hp.get("n_estimators", 200), max_depth=hp["max_depth"], min_samples_leaf=50, max_features="sqrt", n_jobs=nthr, class_weight={0: 1.0, 1: w}, random_state=a.seed); m.fit(Xtr, ytr)
        joblib.dump(m, f"{run_dir}/model.joblib", compress=3); pr = lambda rs, cb=cube: m.predict_proba(tab_matrix(cb, rs, nb))[:, 1]
        info = dict(n_params=int(sum(e.tree_.node_count for e in m.estimators_)), epochs_or_trees=len(m.estimators_), cfg=hp)
    else:
        import xgboost as xgb
        P = dict(tree_method="hist", device=a.device if a.device == "cuda" else "cpu", max_depth=hp["max_depth"], learning_rate=0.1, subsample=1.0, colsample_bytree=0.8, scale_pos_weight=w, objective="binary:logistic", eval_metric="aucpr", nthread=nthr, seed=a.seed)
        dtr = xgb.DMatrix(Xtr, label=ytr); dva = xgb.DMatrix(tab_matrix(cube, va, nb), label=va.y.values); bst = xgb.train(P, dtr, num_boost_round=hp.get("rounds", 1000), evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=50)
        bst.save_model(f"{run_dir}/model.json"); pr = lambda rs, cb=cube: bst.predict(xgb.DMatrix(tab_matrix(cb, rs, nb)), iteration_range=(0, bst.best_iteration + 1))
        info = dict(n_params=int(bst.best_iteration + 1), epochs_or_trees=int(bst.best_iteration + 1), cfg={**hp, **P}, best_val_auprc=float(bst.best_score))
    info["neighbour_features"] = bool(nb); info.setdefault("best_val_auprc", float(average_precision_score(va.y.values, pr(va)))); return pr, info


def write_pred(path, cube, rs, p, split, model, seed):
    t = rs.t.values.astype(np.int64) - cube.t0; f7 = cube.f7f[t, rs.cell_idx.values.astype(np.int64)] if cube.f7f is not None and t.min() >= 0 else np.zeros(len(rs), np.float32)
    pd.DataFrame({"cell_id": cube.cells.cell_id.values[rs.cell_idx.values].astype(np.int32), "cell_idx": rs.cell_idx.values.astype(np.int32), "date": rs.date_t.values, "label_date": rs.label_date.values, "y": rs.y.values,
                  "y_any": rs.y_any.values.astype(np.int8), "had_det_t": rs.had_det_t_forest.values.astype(np.int8), "fd7f": f7.astype(np.int8), "season": rs.season_flag.values.astype(np.int8), "state": rs.state.astype(str).values,
                  "set": rs.set.astype(str).values, "split": split, "model": model, "seed": np.int16(seed), "p": np.asarray(p, dtype=np.float32)}).astype({"state": "category", "set": "category"}).to_parquet(path, index=False)


def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(DEFAULT_HP)); ap.add_argument("--split", default="A"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--hp", default=None)
    ap.add_argument("--run-id", required=True); ap.add_argument("--out", default=f"{ROOT}/runs"); ap.add_argument("--drop", default=None, choices=[None] + list(GROUPS)); ap.add_argument("--only", default=None, choices=[None] + list(ONLY))
    ap.add_argument("--label", default="y_forest", choices=["y_forest", "y_any"]); ap.add_argument("--device", default="cpu"); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--epoch-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--neighbours", action="store_true", help="tabular models: add neighbour fire-history features")
    a = ap.parse_args(argv)
    run_dir = f"{a.out}/{a.run_id}"; os.makedirs(f"{run_dir}/predictions", exist_ok=True); mpath = f"{run_dir}/manifest.json"
    if os.path.exists(mpath) and json.load(open(mpath)).get("complete"): print("already complete:", a.run_id); return
    log = log_to(f"{run_dir}/train.log"); started = time.strftime("%Y-%m-%dT%H:%M:%S"); t0 = time.time(); neural = a.model in ("mlp", "lstm", "cnn", "convlstm", "c1", "c2")
    names = [str(n) for n in np.load(f"{DATA}/scaler_forest_AG_train.npz", allow_pickle=True)["names"]]
    dropped = GROUPS.get(a.drop, []); keep = [n for n in names if n not in dropped and (a.only is None or n in ONLY[a.only])]
    spec = split_dates(a.split); day = lambda d: (pd.Timestamp(d) - pd.Timestamp("2019-01-01")).days
    win = {"train": ("2022-03-01", "2022-03-31"), "validation": ("2022-04-01", "2022-04-15"), "test": ("2022-04-16", "2022-04-30")}
    cube = Cube(DATASET, keep, a.device, t_range=(day("2022-02-15"), day("2022-05-02")) if a.tiny else None, use_torch=neural)
    if a.tiny:                                                                         # pipeline check: 150 cells with most positives, a few weeks
        base = cube.rows[cube.rows.valid_primary.values].copy(); base["y"] = base.y_forest.astype(np.int8); ld = pd.to_datetime(base.label_date)
        cells = base[(ld >= win["train"][0]) & (ld <= win["train"][1])].groupby("cell_idx").y.sum().nlargest(150).index
        rows = {s_: base[base.cell_idx.isin(cells) & (ld >= w_[0]) & (ld <= w_[1])].reset_index(drop=True) for s_, w_ in win.items()}
    else: rows = {s_: pick_rows(cube, spec, s_, a.label) for s_ in ("train", "validation", "test")}
    tr, va = rows["train"], rows["validation"]
    assert a.tiny or tr.label_date.max() < va.label_date.min(), "training labels must end before the validation window"
    npos = int(tr.y.sum()); w = float(np.sqrt((len(tr) - npos) / max(npos, 1))); log(f"{a.run_id}: train rows {len(tr):,} positives {npos:,} w={w:.3f}; val rows {len(va):,}; channels {len(keep)}; device {a.device}")
    fit = fit_neural if neural else fit_tabular; pr, info = fit(a, cube, tr, va, w, run_dir, log)
    for nm, rs in rows.items():
        if nm == "train": continue
        write_pred(f"{run_dir}/predictions/{nm}.parquet", cube, rs, pr(rs), nm, a.model, a.seed); log(f"wrote {nm}: {len(rs):,} rows")
    man = dict(run_id=a.run_id, model=a.model, variant=a.model + (f"__drop_{a.drop}" if a.drop else "") + (f"__only_{a.only}" if a.only else "") + ("__y_any" if a.label == "y_any" else ""), seed=a.seed, split=a.split, hp=info.get("cfg"), n_params=info.get("n_params"),
               epochs_or_trees=info.get("epochs_or_trees"), best_val_auprc=info.get("best_val_auprc"), channels=keep, neighbour_features=info.get("neighbour_features", False), label=a.label, train_rows=len(tr), val_rows=len(va), positive_weight=w,
               wallclock_s=round(time.time() - t0), machine=f"{platform.node()} {platform.platform()} device={a.device}", started=started, ended=time.strftime("%Y-%m-%dT%H:%M:%S"), script_sha256=sha(__file__),
               libraries=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, **({"torch": cube.torch.__version__} if neural else {})), tiny=a.tiny, complete=True)
    json.dump(man, open(mpath, "w"), indent=1); json.dump(vars(a), open(f"{run_dir}/config.json", "w"), indent=1); log("done " + a.run_id)


if __name__ == "__main__":
    main()
