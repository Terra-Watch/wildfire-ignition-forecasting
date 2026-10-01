"""Walk-through of the baseline for notebook 00: raw data, inputs, one fold refitted with the official code, and the
official results beside training-set and thresholded metrics.

Official parts call the project's own functions (dataset, fit-row cap, design weights, model classes, prior-shift
correction, ladder metrics). Parts that are NOT in the official tables are named as such: the training-set metrics and the
thresholded metrics (accuracy, precision, recall, F1), the per-epoch network history and the block permutation importance."""
from __future__ import annotations

import gc
import glob
import json
import os
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.calibration import calibration_curve  # noqa: F401  (kept for notebooks that ask for it)
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, precision_recall_curve, roc_curve

from wildfire import config
from wildfire.evaluation import classification, importance, ladder, metrics as official_metrics, views as official_views
from wildfire.features import neighbourhood as nbmod, weather_windows
from wildfire.labels import weights as label_weights
from wildfire.labels.detections import load_grid
from wildfire.models.fit import Model
from wildfire.models.network import Standardiser
from wildfire.models.network_traced import TracedNetwork
from wildfire.pipelines import baseline_tables, dataset, ladder_fit, rolling_origin

COLOURS = {"logistic": "#6b8f71", "boosting": "#d6532f", "network": "#5b8def"}
LAST_NOTEBOOK_YEAR = 2024            # the notebook never loads 2025, the fresh states or the 2026 season


# ----------------------------------------------------------------------------------------------------------------
# 1. Data extraction (read-only, small samples)
# ----------------------------------------------------------------------------------------------------------------
def raw_detections() -> pd.DataFrame:
    """One raw FIRMS file from the middle of the archive (five days of detections)."""
    files = sorted(glob.glob(str(config.data_root() / "raw/firms*/VIIRS*.csv")))
    print(len(files), "VIIRS files; first:", files[0].split("/")[-1], "| last:", files[-1].split("/")[-1])
    return pd.read_csv(files[len(files) // 2], dtype={"confidence": str})


def filter_summary(raw: pd.DataFrame) -> None:
    """The pipeline's detection filter applied to one file."""
    cfg = config.load("domain")["detections"]
    kept = raw[raw.confidence.isin(cfg["confidence_keep"]) & raw.satellite.isin(cfg["satellites_keep"])].drop_duplicates(cfg["duplicate_key"])
    print(f"one file: {len(raw):,} detections -> {len(kept):,} after filters ({len(kept) / len(raw):.0%} kept)")
    print("confidence kept:", cfg["confidence_keep"], "| satellites kept:", cfg["satellites_keep"])


def grid_overview() -> pd.DataFrame:
    cells, _ = load_grid()
    cells.role.value_counts().plot.bar(title="Grid cells by role", ylabel="cells", rot=0)
    plt.show()
    return cells.head()


def cell_day_overview() -> None:
    """Cell-day table: head, per-year counts and onsets by month."""
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv")
    cd = cd[cd.year <= LAST_NOTEBOOK_YEAR]
    display(cd.head())
    display(cd.groupby("year").agg(cell_days=("n", "size"), onsets=("onset", "sum")))
    cd[cd.onset].assign(month=lambda d: pd.to_datetime(d.day, unit="D").dt.month).groupby("month").size().plot.bar(
        title="Fire onsets by month (2019-2024)", ylabel="onsets", rot=0)
    plt.show()


def labelled_rows_overview() -> None:
    """Labelled rows of the fit cells: head and the label balance per year."""
    rows = pd.read_csv(config.output_dir("labels") / "rows.csv", usecols=["cell", "day", "label", "kind", "state", "year", "group", "band"],
                       dtype={"kind": "category", "state": "category", "group": "category", "band": "category"})
    rows = rows[(rows.group == "train") & (rows.year <= LAST_NOTEBOOK_YEAR)]
    display(rows.head())
    bal = rows.groupby("year").label.agg(rows="size", onsets="sum")
    bal["share_positive"] = bal.onsets / bal.rows
    display(bal)


def architecture(name: str) -> None:
    """Draw one of the architecture diagrams (docs/figures/draw_architecture.py)."""
    draw = runpy.run_path(str(config.REPO_ROOT / "docs/figures/draw_architecture.py"))
    draw[name]()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------
# 2-6. One fold with the official code
# ----------------------------------------------------------------------------------------------------------------
class Walkthrough:
    """State of one rolling-origin fold: fit on the years before `test_year`, evaluate on its January-July rows."""

    def __init__(self, level: str = "enriched", seed: int = 0):
        self.smoke = os.environ.get("NB_SMOKE") == "1"          # tiny test of the notebook itself; the real run leaves it off
        self.test_year = 2022 if self.smoke else LAST_NOTEBOOK_YEAR
        self.val_year = self.test_year - 1
        self.level, self.seed = level, seed
        self.fit_cap = 60_000 if self.smoke else None            # None = the official fit-row cap
        self.train_sample = 20_000 if self.smoke else 200_000
        self.tuned = json.loads((config.output_dir("results") / "tuned.json").read_text())
        self.fresh: dict = {}
        if self.smoke:
            print("SMOKE run: tiny row caps, 2022 fold. Differences from the saved official run are expected.")

    # -- inputs ---------------------------------------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        self.ds = dataset.load_dataset()
        self.t = self.ds.table
        self.features = self.ds.tabular_columns["boosting"]
        self.enriched = dataset.enriched_names()
        print(f"{len(self.t):,} labelled rows; {int(self.ds.complete.sum()):,} complete ({self.ds.complete.mean():.1%})")
        print(f"inputs: {len(self.features)} tabular + {len(self.enriched)} enriched = {len(self.features) + len(self.enriched)} columns at the enriched level")
        return self.t[["cell", "day", "year", "state", "label"] + self.features[:8]].head()

    def enriched_examples(self) -> None:
        win, nb = self.ds.arrays["windows"], self.ds.arrays["neighbourhood"]
        print("weather windows array:", win.shape, "| neighbourhood array:", nb.shape)
        display(pd.DataFrame(np.asarray(win[:5, :6]), columns=weather_windows.window_names()[:6]))
        display(pd.DataFrame(np.asarray(nb[:5]), columns=nbmod.column_names()).round(3))

    def _sample(self, n: int) -> pd.DataFrame:
        ok = np.flatnonzero(self.ds.complete & (self.t.year.values <= self.test_year))
        return self.t.iloc[np.random.default_rng(self.seed).choice(ok, min(n, len(ok)), replace=False)]

    def plot_distributions(self) -> None:
        s = self._sample(200_000)
        fig, axes = plt.subplots(2, 3, figsize=(11, 5))
        for ax, f in zip(axes.ravel(), ["vpd", "fwi", "rain7", "dsr", "sw1_anom", "wc_20"]):
            lo, hi = s[f].quantile([0.01, 0.99])
            for lab, c in ((0, "#8a8a8a"), (1, "#d6532f")):
                ax.hist(s.loc[s.label == lab, f].clip(lo, hi), bins=40, density=True, alpha=0.6, color=c, label=f"label {lab}")
            ax.set_title(f)
        axes[0, 0].legend()
        plt.tight_layout()
        plt.show()

    def plot_correlations(self) -> None:
        corr = self._sample(100_000)[self.features].corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title("Tabular input correlations")
        ax.set_xticks(range(len(self.features)))
        ax.set_xticklabels(self.features, rotation=90, fontsize=6)
        ax.set_yticks(range(len(self.features)))
        ax.set_yticklabels(self.features, fontsize=6)
        fig.colorbar(im)
        plt.show()

    # -- the split exactly as the official fold builds it ---------------------------------------------------------
    def make_split(self) -> None:
        t, year = self.t, self.test_year
        self.fit_idx = dataset.fit_rows(self.ds, year, self.fit_cap)
        self.val_idx = dataset.validation_rows(self.ds, year)
        self.y_fit = t.label.values[self.fit_idx]
        self.w_fit = ladder_fit.training_weights(t, self.fit_idx)
        split = pd.DataFrame({"rows": [len(self.fit_idx), len(self.val_idx)],
                              "onsets": [int(self.y_fit.sum()), int(t.label.values[self.val_idx].sum())],
                              "years": [f"{t.year.values[self.fit_idx].min()}-{t.year.values[self.fit_idx].max()}", f"{year} (Jan-Jul)"]}, index=["fit", "test"])
        split["share positive"] = (split.onsets / split.rows).round(4)
        display(split)
        display(pd.DataFrame({"mean design weight": pd.Series(self.w_fit).groupby(self.y_fit).mean()}).rename(
            index={0: "negative rows", 1: "onset rows"}).round(2))
        rng = np.random.default_rng(self.seed)
        self.sub_pos = np.sort(rng.choice(len(self.fit_idx), min(self.train_sample, len(self.fit_idx)), replace=False))
        self.sub_idx = self.fit_idx[self.sub_pos]
        rate = label_weights.rates_before(t, year)                      # per-state onset rate before the test year
        self.pi_sub = np.array([rate[s] for s in t.state.values[self.sub_idx]])
        self.y_sub, self.w_sub = self.y_fit[self.sub_pos], self.w_fit[self.sub_pos]
        self.cols = ladder_fit.columns_for(self.ds, "logistic_with_burn")
        self.xv = ladder_fit.validation_file(self.ds, self.val_idx, self.cols, self.level, year)   # test-year inputs, memory-mapped

    def plot_tuning(self) -> None:
        trials = pd.read_csv(config.output_dir("results") / "tuning_trials.csv")
        print(json.dumps(self.tuned, indent=1))
        mean = trials.groupby(["kind", "trial"]).log_loss.mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, kind in zip(axes, ["boosting", "network"]):
            m = mean[mean.kind == kind]
            best = m.loc[m.log_loss.idxmin()]
            ax.bar(m.trial, m.log_loss, color=["#d6532f" if tr == best.trial else "#9aa5b1" for tr in m.trial])
            ax.set_ylim(m.log_loss.min() * 0.98, m.log_loss.max() * 1.01)
            ax.set_title(f"{kind}: mean log loss, 2021-2024")
            ax.set_xlabel("trial")
        plt.tight_layout()
        plt.show()

    # -- models, fitted with the official classes and steps -------------------------------------------------------
    def _store(self, name: str, model, test: np.ndarray, train: np.ndarray) -> None:
        self.fresh[name] = dict(test=test, train=train, prev=model.fit_prevalence)

    def fit_logistic(self) -> None:
        year = self.test_year
        x = dataset.matrix(self.ds, self.fit_idx, self.cols, self.level, year)
        self.x_sub = x[self.sub_pos].copy()                            # raw training sample, reused by every model
        kind, params = ladder_fit.settings_for("logistic_with_burn", self.tuned)
        model = Model(kind, params).fit(x, self.y_fit)   # as ladder_fit.fit_standard
        del x
        gc.collect()
        self._store("logistic_with_burn@" + self.level, model, dataset.predict_chunks(model.predict, self.xv), model.predict(self.x_sub))
        print(kind, params, "| inputs:", self.x_sub.shape[1])

    def fit_boosting(self) -> None:
        kind, params = ladder_fit.settings_for("boosting", self.tuned)
        x = dataset.matrix(self.ds, self.fit_idx, self.cols, self.level, self.test_year)
        self.boost = Model(kind, params).fit(x, self.y_fit)
        del x
        gc.collect()
        self._store("boosting@" + self.level, self.boost, dataset.predict_chunks(self.boost.predict, self.xv), self.boost.predict(self.x_sub))
        self.boost_params = params

    def plot_boosting_curve(self) -> None:
        """Log loss as trees are added (sampled rate, unweighted)."""
        vs = np.sort(np.random.default_rng(self.seed).choice(len(self.val_idx), min(60_000, len(self.val_idx)), replace=False))
        self.x_vs, y_vs = np.asarray(self.xv[vs]), self.t.label.values[self.val_idx][vs]
        ss = np.arange(min(60_000, len(self.sub_pos)))
        stages = list(range(0, self.boost_params["max_iter"], 10))
        tr = [log_loss(self.y_sub[ss], p[:, 1]) for i, p in enumerate(self.boost.m.staged_predict_proba(self.x_sub[ss])) if i in stages]
        te = [log_loss(y_vs, p[:, 1]) for i, p in enumerate(self.boost.m.staged_predict_proba(self.x_vs)) if i in stages]
        plt.plot(stages, tr, label="train sample")
        plt.plot(stages, te, label=f"test {self.test_year} sample")
        plt.xlabel("trees")
        plt.ylabel("log loss (sampled rate, unweighted)")
        plt.title("Boosting: loss as trees are added")
        plt.legend()
        plt.show()

    def fit_network(self) -> pd.DataFrame:
        """Seed 0 of the official network fit (`ladder_fit.fit_network`), keeping the per-epoch history (not official)."""
        t, year = self.t, self.test_year
        stop = t.year.values[self.fit_idx] == year - 1
        x = dataset.matrix(self.ds, self.fit_idx[~stop], self.cols, self.level, year)
        std = Standardiser(x)
        std.apply(x)
        xs = std.apply(dataset.matrix(self.ds, self.fit_idx[stop], self.cols, self.level, year))
        keep = ~stop[self.sub_pos]
        probe_x = std.apply(self.x_sub[keep].copy())
        net = TracedNetwork(self.tuned["network"], seed=0).fit(x, self.y_fit[~stop], self.w_fit[~stop], xs, self.y_fit[stop], self.w_fit[stop],
                                                              probe_x, self.y_sub[keep], self.w_sub[keep])
        del x, xs
        gc.collect()
        self.net = net
        self._store("tabular_network@" + self.level + "#0", net, dataset.predict_chunks(net.predict, self.xv, transform=std.apply),
                    net.predict(std.apply(self.x_sub.copy())))
        self.net_history = pd.DataFrame(net.history)
        return self.net_history.round(4)

    def plot_network_curve(self) -> None:
        h = self.net_history
        plt.plot(h.epoch, h.train_loss, label="train sample (weighted)")
        plt.plot(h.epoch, h.stop_year_loss, label=f"early-stopping year {self.test_year - 1}")
        plt.xlabel("epoch")
        plt.ylabel("design-weighted log loss")
        plt.title("Network: loss per epoch (best epoch kept)")
        plt.legend()
        plt.show()

    # -- official results -----------------------------------------------------------------------------------------
    def _folds(self) -> dict:
        if not hasattr(self, "folds"):
            self.folds = {y: rolling_origin.load_year(y) for y in range(2021, self.test_year + 1)}
        return self.folds

    def ladder_table(self, year: int) -> pd.DataFrame:
        """Official metrics of every arm for one validation year (network seeds averaged), from the saved predictions."""
        f, prev = self._folds()[year]
        arms = baseline_tables.prediction_columns(f)
        corrected = {a: ladder.corrected_probabilities(f["p_" + a].values, f.pi_true.values, prev[a]) for a in arms}
        scores = {a: (f["score_" + a].values if "score_" + a in f else f["p_" + a].values) for a in arms}
        tab = pd.DataFrame([dict(arm=a, **ladder.arm_row(f, scores[a], corrected[a], f.weight.values)) for a in arms])
        tab["model"] = tab.arm.str.replace(r"#\d+$", "", regex=True)
        return tab.groupby("model", sort=False).mean(numeric_only=True)

    def official_ladder(self) -> pd.DataFrame:
        cols = ["ap", "auc_spatial_within_day", "auc_temporal_within_cell", "log_loss", "brier", "capture_5pct", "lift_5pct"]
        return self.ladder_table(self.test_year)[cols].round(4)

    def plot_by_year(self) -> None:
        years = list(range(2021, self.test_year + 1))
        by_year = pd.concat({y: self.ladder_table(y) for y in years}, names=["year", "model"]).reset_index()
        focus = [m for m in ["climatology_cell", "logistic_with_burn@" + self.level, "boosting@" + self.level, "tabular_network@" + self.level]
                 if m in by_year.model.unique()]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for m in focus:
            d = by_year[by_year.model == m]
            axes[0].plot(d.year, d.ap, "o-", label=m)
            axes[1].plot(d.year, d.log_loss, "o-", label=m)
        axes[0].set_title("Average precision by validation year")
        axes[1].set_title("Corrected log loss by validation year")
        for ax in axes:
            ax.set_xticks(years)
            ax.set_xlabel("validation year")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    # -- train / validation / test --------------------------------------------------------------------------------
    def compare(self) -> None:
        """Build the three splits for the three models and score them. Official metrics and the NOT OFFICIAL thresholded
        metrics are kept in two separate tables."""
        lv = self.level
        self.models = ["logistic_with_burn@" + lv, "boosting@" + lv, "tabular_network@" + lv + "#0"]
        self.short = dict(zip(self.models, ["logistic", "boosting", "network"]))
        f_val, prev_val = self._folds()[self.val_year]
        f_te, prev_te = self._folds()[self.test_year]
        self.f_te, self.prev_te = f_te, prev_te

        def held_out(f, prev, arm):
            return dict(y=f.label.values, raw=f["p_" + arm].values, p=ladder.corrected_probabilities(f["p_" + arm].values, f.pi_true.values, prev[arm]), w=f.weight.values)

        self.splits = {}
        for a in self.models:
            tr = self.fresh[a]
            self.splits[a] = {"train": dict(y=self.y_sub, raw=tr["train"], p=ladder.corrected_probabilities(tr["train"], self.pi_sub, tr["prev"]), w=self.w_sub),
                              "val": held_out(f_val, prev_val, a), "test": held_out(f_te, prev_te, a)}
        self.thr = {a: classification.best_threshold(self.splits[a]["val"]["y"], self.splits[a]["val"]["p"], self.splits[a]["val"]["w"]) for a in self.models}
        off, extra = {}, {}
        for a in self.models:
            for s, d in self.splits[a].items():
                off[(self.short[a], s)] = dict(log_loss=official_metrics.log_loss(d["y"], d["p"], w=d["w"]), brier=official_metrics.brier(d["y"], d["p"], w=d["w"]),
                                               ap=float(average_precision_score(d["y"], d["raw"])), auc=official_views.auc(d["y"], d["raw"]))
                extra[(self.short[a], s)] = classification.threshold_metrics(d["y"], d["p"], d["w"], self.thr[a])
        self.official_metrics = pd.DataFrame(off).T
        self.extra_metrics = pd.DataFrame(extra).T
        for df in (self.official_metrics, self.extra_metrics):
            df.index.names = ["model", "split"]
        ref = self.splits[self.models[0]]
        self.always_no = {s: classification.always_no_accuracy(ref[s]["y"], ref[s]["w"]) for s in ref}

    def show_official_metrics(self) -> pd.DataFrame:
        print("OFFICIAL functions (design-weighted, prior-shift-corrected log loss and Brier; AP and AUC on the raw scores).")
        print("The train rows are a sample of this fold's own fit rows and are not part of the official tables.")
        return self.official_metrics.round(4)

    def show_extra_metrics(self) -> pd.DataFrame:
        print("NOT OFFICIAL: thresholded metrics added for this notebook. Threshold per model (corrected probability, chosen on validation):",
              {self.short[a]: round(v, 4) for a, v in self.thr.items()})
        print("Weighted accuracy of always saying 'no fire':", {k: round(v, 4) for k, v in self.always_no.items()})
        return self.extra_metrics.round(4)

    def plot_train_val_test(self) -> pd.DataFrame:
        both = pd.concat([self.official_metrics, self.extra_metrics], axis=1)
        order = list(self.short.values())
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, (met, official) in zip(axes.ravel(), [("log_loss", True), ("ap", True), ("auc", True), ("f1", False), ("precision", False), ("recall", False)]):
            both[met].unstack("split")[["train", "val", "test"]].loc[order].plot.bar(ax=ax, rot=0, color=["#9aa5b1", "#5b8def", "#d6532f"], legend=False)
            ax.set_title(met + ("" if official else "  (not official)"))
            ax.set_xlabel("")
        axes[0, 0].legend(["train sample", "validation", "test"])
        plt.tight_layout()
        plt.show()
        tr, te = both.xs("train", level="split"), both.xs("test", level="split")
        gap = (tr[["log_loss", "ap", "f1"]] - te[["log_loss", "ap", "f1"]]).round(4)
        gap.columns = ["train - test log loss", "train - test AP", "train - test F1 (not official)"]
        return gap

    def plot_curves(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for a in self.models:
            d, m = self.splits[a]["test"], self.short[a]
            fpr, tpr, _ = roc_curve(d["y"], d["raw"])
            axes[0].plot(fpr, tpr, color=COLOURS[m], label=f"{m} (AUC {official_views.auc(d['y'], d['raw']):.3f})")
            pr, rc, _ = precision_recall_curve(d["y"], d["raw"])
            axes[1].plot(rc, pr, color=COLOURS[m], label=f"{m} (AP {average_precision_score(d['y'], d['raw']):.3f})")
            rel = ladder.reliability(d["y"], d["p"], d["w"], config.load("baseline")["reporting"]["reliability_bins"])
            axes[2].plot(rel.mean_predicted, rel.observed_rate, "o-", color=COLOURS[m], label=m)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set(title=f"ROC (test {self.test_year})", xlabel="false positive rate", ylabel="true positive rate")
        axes[1].set(title=f"Precision-recall, sampled rows (test {self.test_year})", xlabel="recall", ylabel="precision")
        hi = float(max(axes[2].get_xlim()[1], axes[2].get_ylim()[1]))
        axes[2].plot([0, hi], [0, hi], "k--", lw=0.8)
        axes[2].set(title=f"Reliability, corrected probabilities (test {self.test_year})", xlabel="mean predicted probability", ylabel="weighted observed rate")
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion(self) -> None:
        """NOT OFFICIAL: design-weighted confusion matrices at each model's threshold, as the share of each true class."""
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, a in zip(axes, self.models):
            d = self.splits[a]["test"]
            cm = confusion_matrix(d["y"], d["p"] >= self.thr[a], sample_weight=d["w"], normalize="true")
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=11)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred no fire", "pred fire"])
            ax.set_yticklabels(["no fire", "fire"])
            ax.set_title(f"{self.short[a]} (threshold {self.thr[a]:.3f})")
        plt.tight_layout()
        plt.show()

    def topk_table(self) -> pd.DataFrame:
        shares = config.load("baseline")["reporting"]["top_k_share"]
        return pd.DataFrame({self.short[a]: ladder.top_k_capture(self.splits[a]["test"]["y"], self.splits[a]["test"]["raw"], shares) for a in self.models}).T.round(3)

    def plot_importance(self) -> None:
        """NOT OFFICIAL: block permutation importance of the boosting model on a test sample."""
        n_feat, n_enr = len(self.features), len(self.enriched)
        n_win = len(weather_windows.window_names())
        blocks = {f: [i] for i, f in enumerate(self.features)}
        blocks.update({"[block] weather windows": list(range(n_feat, n_feat + n_win)),
                       "[block] anomalies": list(range(n_feat + n_win, n_feat + n_enr - 12)),
                       "[block] neighbourhood": list(range(n_feat + n_enr - 12, n_feat + n_enr))})
        y = self.t.label.values[self.val_idx]
        pick = np.sort(np.random.default_rng(self.seed).choice(len(y), min(60_000, len(y)), replace=False))
        base, drops = importance.block_permutation_drops(self.boost.predict, np.asarray(self.xv[pick]), y[pick], blocks, self.seed)
        imp = pd.Series(drops).sort_values().tail(14)
        imp.plot.barh(title=f"Boosting: drop in AP when shuffled (test {self.test_year} sample, base AP {base:.3f}; not official)",
                      color=["#7b5ea7" if i.startswith("[block]") else "#d6532f" for i in imp.index])
        plt.xlabel("AP drop")
        plt.show()

    def match_check(self) -> pd.DataFrame:
        """The refit's test-year predictions against the saved official ones."""
        out = []
        for a in self.models:
            off, mine = self.f_te["p_" + a].values, self.fresh[a]["test"]
            out.append(dict(model=self.short[a], rows=len(off), max_abs_diff=float(np.max(np.abs(off - mine))), identical=bool(np.allclose(off, mine, atol=1e-6)),
                            fit_prevalence_saved=self.prev_te[a], fit_prevalence_refit=self.fresh[a]["prev"]))
        dataset.clear_cache()
        if self.smoke:
            print("SMOKE run: the row cap differs from the official one, so differences are expected.")
        return pd.DataFrame(out)

    def summary(self) -> None:
        both = pd.concat([self.official_metrics, self.extra_metrics], axis=1)
        te, tr = both.xs("test", level="split"), both.xs("train", level="split")
        best = te.log_loss.idxmin()
        print(f"Test {self.test_year}: lowest corrected log loss is {best} ({te.log_loss[best]:.4f}).")
        print("Test AP: " + ", ".join(f"{m} {te.ap[m]:.3f}" for m in te.index))
        print("NOT OFFICIAL, test F1 / precision / recall: " + "; ".join(f"{m} {te.f1[m]:.3f} / {te.precision[m]:.3f} / {te.recall[m]:.3f}" for m in te.index))
        print("NOT OFFICIAL, test accuracy: " + ", ".join(f"{m} {te.accuracy[m]:.4f}" for m in te.index) + f" (always 'no fire': {self.always_no['test']:.4f}).")
        print("Train - test log loss: " + ", ".join(f"{m} {tr.log_loss[m] - te.log_loss[m]:+.4f}" for m in te.index) + " (negative = better on train).")
