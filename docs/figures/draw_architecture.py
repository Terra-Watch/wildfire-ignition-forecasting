"""Diagrams of the baseline: data flow, model ladder, network, boosting, rolling-origin folds.
Plain matplotlib; every function returns a figure. Run this file to write the PNGs next to it."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"
COL = {"data": "#2f6fb0", "feat": "#2a9d8f", "model": "#e07a2f", "eval": "#7b5ea7", "guard": "#c0392b"}
LIGHT = {k: v + "22" for k, v in COL.items()}


def _box(ax, x, y, w, h, title, lines, kind, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.02", fc=LIGHT[kind], ec=COL[kind], lw=1.6))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=fs + 1, fontweight="bold", color=COL[kind])
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=fs - 0.5, color=INK, linespacing=1.45)


def _arrow(ax, p, q, color=MUTED, rad=0.0, lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7.0)); ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    w, h = 3.15, 2.2
    xs = [0.2 + i * 3.45 for i in range(4)]
    top = [("1  Fire detections", ["NASA FIRMS, VIIRS", "keep confidence n / h", "satellites N, N20", "drop duplicates"], "data"),
           ("2  5 km grid", ["study area cut into cells", "roles: train, held-out UK,", "unused (AP), reserve, buffer", "only train cells are fitted"], "data"),
           ("3  Cell-days", ["detections placed in cells", "onset = a detection with", "none in the 3 days before", "onsets are the positives"], "data"),
           ("4  Labelled rows", ["each onset (label 1) +", "20 matched negatives (0)", "same state and year, in season,", "no detection in days D-3..D"], "data")]
    bot = [("5  Inputs known by day D", ["tabular: 42 columns", "weather, fire-weather, terrain,", "land cover, human, season, burn", "enriched: +209 more columns"], "feat"),
           ("6  Model ladder", ["7 arms, simple to complex", "rate, logistic, boosting, network", "one tuned setting for boosting", "and for the network"], "model"),
           ("7  Rolling-origin folds", ["fit on earlier years, predict next", "validation years 2021 ... 2025", "prior-shift: 4.76% -> 1.47%", "metrics within day and within cell"], "eval"),
           ("8  Guards and tables", ["8 guards must pass first", "record bound to clean code + arrays", "then metrics, contrasts, reliability", "spec frozen at commit 99c867c"], "guard")]
    for i, (t, l, k) in enumerate(top):
        _box(ax, xs[i], 4.6, w, h, t, l, k)
        if i: _arrow(ax, (xs[i] - 0.3, 5.7), (xs[i], 5.7))
    for i, (t, l, k) in enumerate(bot):
        _box(ax, xs[i], 1.1, w, h, t, l, k)
        if i: _arrow(ax, (xs[i] - 0.3, 2.2), (xs[i], 2.2))
    ym = 4.0
    ax.plot([xs[3] + w / 2, xs[3] + w / 2], [4.55, ym], color=MUTED, lw=1.8); ax.plot([xs[3] + w / 2, xs[0] + w / 2], [ym, ym], color=MUTED, lw=1.8)
    _arrow(ax, (xs[0] + w / 2, ym), (xs[0] + w / 2, 3.35))
    ax.text(7, 7.35, "Baseline pipeline: from raw fire detections to scored predictions", ha="center", fontsize=14, fontweight="bold", color=INK)
    for k, (lab, x) in enumerate([("data", 3.3), ("features", 5.4), ("models", 7.5), ("evaluation", 9.6), ("safeguards", 11.7)]):
        c = ["data", "feat", "model", "eval", "guard"][k]
        ax.add_patch(Rectangle((x, 0.45), 0.22, 0.22, fc=COL[c], ec="none")); ax.text(x + 0.32, 0.56, lab, va="center", fontsize=9.5, color=INK)
    return fig


def plot_ladder():
    arms = [("climatology_cell", "historical rate per cell", 1, "one rate per cell"),
            ("landcover_season", "logistic regression, C = 1", 10, "10 weights + 1"),
            ("fwi_season", "logistic regression, C = 0.01", 8, "8 weights + 1"),
            ("logistic_weather_static", "logistic regression, C = 0.01", 40, "40 weights + 1"),
            ("logistic_with_burn", "logistic regression, C = 0.01", 42, "42 weights + 1"),
            ("boosting", "gradient-boosted trees", 42, "<= 500 trees, <= 15,500 leaves"),
            ("tabular_network", "MLP 42-256-256-1", 42, "77,057 weights")]
    cols = ["#9aa5b1", "#7aa6d6", "#7aa6d6", "#2f6fb0", "#2f6fb0", "#e07a2f", "#d6532f"]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    y = np.arange(len(arms))[::-1]
    ax.barh(y, [a[2] for a in arms], color=cols, height=0.62)
    for yi, (n, desc, k, par) in zip(y, arms):
        ax.text(-1, yi, n, ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(k + 1, yi + 0.12, f"{k} input" + ("s" if k > 1 else ""), va="center", fontsize=9.5, color=INK)
        ax.text(k + 1, yi - 0.17, f"{desc}  |  {par}", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 80); ax.set_yticks([]); ax.set_xlabel("input columns at the tabular level (the enriched level adds 209 more to the last four arms)")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("The model ladder: each arm adds information or flexibility to the one above", fontsize=13, fontweight="bold", color=INK, loc="left", x=-0.28)
    ax.annotate("", xy=(0.99, 0.05), xytext=(0.99, 0.95), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(0.97, 0.5, "more flexible", rotation=90, transform=ax.transAxes, ha="right", va="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.12)
    return fig


def plot_network():
    fig, ax = plt.subplots(figsize=(13, 6.0)); ax.set_xlim(0, 13); ax.set_ylim(-1.1, 6.2); ax.axis("off")
    layers = [("Input", "42 columns", 42, "#9aa5b1", None), ("Hidden 1", "Linear(42 -> 256)\nReLU, Dropout 0.1", 256, "#e07a2f", "11,008 weights"),
              ("Hidden 2", "Linear(256 -> 256)\nReLU, Dropout 0.1", 256, "#e07a2f", "65,792 weights"), ("Output", "Linear(256 -> 1)\nsigmoid", 1, "#d6532f", "257 weights")]
    x = 0.9
    pos = []
    for name, desc, n, c, par in layers:
        hgt = 0.7 + 2.6 * np.sqrt(n / 256) if n > 1 else 0.7
        ax.add_patch(FancyBboxPatch((x, 2.9 - hgt / 2), 1.5, hgt, boxstyle="square,pad=0.02", fc=c + "33", ec=c, lw=1.8))
        ax.text(x + 0.75, 2.9, f"{n}" if n != 42 else "42", ha="center", va="center", fontsize=15, fontweight="bold", color=c)
        ax.text(x + 0.75, 2.9 + hgt / 2 + 0.3, name, ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(x + 0.75, 2.9 - hgt / 2 - 0.35, desc, ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.4)
        pos.append((x, hgt, par)); x += 3.2
    for (x0, h0, _), (x1, h1, par) in zip(pos[:-1], pos[1:]):
        _arrow(ax, (x0 + 1.55, 2.9), (x1 - 0.05, 2.9), lw=2.2)
        ax.text((x0 + 1.5 + x1) / 2, 3.15, par, ha="center", fontsize=9.5, color=MUTED)
    ax.text(x - 3.2 + 2.0, 2.9, "p(fire)", fontsize=12, fontweight="bold", color="#d6532f", va="center")
    ax.text(6.5, 5.95, "Tabular network: a small multilayer perceptron (77,057 weights)", ha="center", fontsize=13, fontweight="bold", color=INK)
    card = "Training: design-weighted binary cross-entropy  |  AdamW, lr 0.0003, weight decay 0.0001  |  batch 4096, <= 20 epochs\nEarly stopping: patience 2 on the last fit year, best epoch kept  |  inputs standardised with fit-row mean and std"
    ax.text(6.5, -0.65, card, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.6, bbox=dict(boxstyle="square,pad=0.5", fc="#f4f6f8", ec=GRID))
    return fig


def _tree(ax, x, y, depth, spread, color):
    if depth == 0:
        ax.add_patch(Circle((x, y), 0.07, fc=color, ec="none")); return
    ax.add_patch(Circle((x, y), 0.07, fc="white", ec=color, lw=1.4))
    for dx in (-spread, spread):
        ax.plot([x, x + dx], [y, y - 0.45], color=color, lw=1.2); _tree(ax, x + dx, y - 0.45, depth - 1, spread / 2, color)


def plot_boosting():
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    for i, (cx, lab) in enumerate([(1.6, "tree 1"), (4.0, "tree 2"), (6.4, "..."), (8.8, "tree 500")]):
        if lab != "...": _tree(ax, cx, 3.8, 3, 0.55, "#e07a2f"); ax.text(cx, 1.95, lab, ha="center", fontsize=10.5, fontweight="bold", color=INK)
        else: ax.text(cx, 3.2, "...", ha="center", fontsize=26, color=MUTED)
        if i < 3: ax.text(cx + 1.2, 3.1, "+", ha="center", fontsize=20, color=MUTED)
    ax.text(1.6, 1.55, "each tree corrects the errors\nof the trees before it", ha="center", va="top", fontsize=9, color=MUTED)
    _arrow(ax, (9.9, 3.1), (10.6, 3.1), lw=2.2)
    ax.add_patch(FancyBboxPatch((10.7, 2.5), 2.1, 1.2, boxstyle="square,pad=0.02", fc="#d6532f33", ec="#d6532f", lw=1.8))
    ax.text(11.75, 3.1, "sigmoid(\n0.05 x sum of trees)", ha="center", va="center", fontsize=10, color=INK, linespacing=1.4)
    ax.text(6.5, 4.95, "Boosting: up to 500 small trees added one at a time (scikit-learn HistGradientBoosting)", ha="center", fontsize=13, fontweight="bold", color=INK)
    rows = [("max_depth", "8"), ("learning_rate", "0.05"), ("max_iter (trees)", "500"), ("min_samples_leaf", "100"), ("l2_regularization", "0"), ("leaves per tree", "31 (default)")]
    for k, (a, b) in enumerate(rows):
        xx = 0.9 + k * 2.0
        ax.add_patch(FancyBboxPatch((xx - 0.05, 0.15), 1.85, 0.9, boxstyle="square,pad=0.02", fc="#f4f6f8", ec=GRID))
        ax.text(xx + 0.875, 0.82, a, ha="center", fontsize=8.5, color=MUTED); ax.text(xx + 0.875, 0.45, b, ha="center", fontsize=12.5, fontweight="bold", color="#e07a2f")
    return fig


def plot_folds():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    years = list(range(2019, 2026))
    for r, v in enumerate(range(2021, 2026)):
        yy = 4 - r
        for y in years:
            c = "#2f6fb0" if y < v else ("#e07a2f" if y == v else "#e9edf1")
            ax.add_patch(Rectangle((y - 2019, yy - 0.38), 0.94, 0.76, fc=c, ec="white"))
        ax.text(-0.15, yy, f"validate {v}", ha="right", va="center", fontsize=10, color=INK)
    for y in years: ax.text(y - 2019 + 0.47, 4.75, str(y), ha="center", fontsize=10, color=INK)
    ax.set_xlim(-1.3, 7); ax.set_ylim(-1.1, 5.2); ax.axis("off")
    ax.add_patch(Rectangle((1.0, -1.0), 0.3, 0.25, fc="#2f6fb0", ec="none")); ax.text(1.4, -0.88, "fit on", va="center", fontsize=9.5)
    ax.add_patch(Rectangle((2.6, -1.0), 0.3, 0.25, fc="#e07a2f", ec="none")); ax.text(3.0, -0.88, "predict (January-July)", va="center", fontsize=9.5)
    ax.add_patch(Rectangle((5.0, -1.0), 0.3, 0.25, fc="#e9edf1", ec="none")); ax.text(5.4, -0.88, "not used", va="center", fontsize=9.5)
    ax.set_title("Rolling-origin evaluation: always fit on the past, predict the next year", fontsize=13, fontweight="bold", color=INK, loc="left", x=-0.02)
    return fig


ALL = {"01_pipeline": plot_pipeline, "02_ladder": plot_ladder, "03_network": plot_network, "04_boosting": plot_boosting, "05_folds": plot_folds}

if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    for name, fn in ALL.items():
        fn().savefig(here / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        print("wrote", name)
