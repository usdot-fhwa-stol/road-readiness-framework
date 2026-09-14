#!/usr/bin/env python3
"""Generate publication-quality comparison figures from lane-eval results.

Driven entirely by the per-image CSVs in ``outputs/results/per_image/``
(``<model>_<dataset>_per_image.csv``). Runs are discovered from those files and
every reported number — means, error bars, distributions — is derived from the
per-image rows, so all four figures are internally consistent (e.g. the mean
diamond on a box plot is exactly the bar height).

NOTE: these are macro (per-image) averages. They differ slightly from the
micro-averaged (pixel-pooled) metrics in the aggregate ``*.json`` files — up to
~0.06 F1 on CULane — because the JSON pools pixels across the whole dataset
while these weight every image equally.

Produces four complementary figures, each saved as both PNG (300 dpi) and PDF:

    fig1_grouped_bars   grouped bars per metric, datasets on x, error bars = std
    fig2_boxplots       per-image metric distributions (the honest "error graph")
    fig3_radar          per-model profile across the 4 mean metrics
    fig4_precision_recall   P-R tradeoff scatter, one point per (model, dataset)

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --results-dir outputs/results --out-dir outputs/results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics we plot, in display order, mapped to their CSV / JSON keys.
METRICS = [
    ("lane_iou", "IoU"),
    ("lane_f1", "F1"),
    ("lane_precision", "Precision"),
    ("lane_recall", "Recall"),
]

# Stable, readable dataset + model ordering and pretty labels.
DATASET_ORDER = ["bdd100k_lane", "culane", "curvelanes", "tusimple"]
DATASET_LABELS = {
    "bdd100k_lane": "BDD100K",
    "culane": "CULane",
    "curvelanes": "CurveLanes",
    "tusimple": "TuSimple",
}
MODEL_ORDER = ["yolopx", "hybridnets", "scnn", "clrernet"]
MODEL_LABELS = {
    "yolopx": "YOLOPX", "hybridnets": "HybridNets",
    "scnn": "SCNN", "clrernet": "CLRerNet",
}
MODEL_COLORS = {
    "yolopx": "#1f77b4", "hybridnets": "#ff7f0e",
    "scnn": "#2ca02c", "clrernet": "#d62728",
}

# Per-image CSVs may end in either suffix, depending on the run that wrote them.
CSV_SUFFIXES = ["_per_image_metrics", "_per_image"]

# Where each model's per-image CSVs live by default, so ``--models all`` can
# pull every model onto one chart without the caller naming directories.
DEFAULT_RESULTS_DIRS = [
    Path("outputs/results"),                # yolopx, hybridnets
    Path("outputs_coremetrics_cs/results"),  # scnn, clrernet
]


def load_results(results_dirs):
    """Load per-image CSVs from one or more dirs and derive per-run aggregates.

    Runs are discovered from ``<dir>/per_image/*.csv`` (either ``_per_image.csv``
    or ``_per_image_metrics.csv``). ``model``/``dataset`` are taken from the CSV
    columns when present, else parsed from the filename ``<model>_<dataset>``
    (model = first token, so dataset underscores like ``bdd100k_lane`` survive).
    Every aggregate is computed from the per-image rows. If the same
    (model, dataset) appears in two dirs, the later directory wins.

    Returns
    -------
    agg : dict[(model, dataset)] -> dict with per-image metric means + count
    per_image : dict[(model, dataset)] -> DataFrame of per-image metric rows
    """
    metric_keys = [k for k, _ in METRICS]
    agg: dict[tuple[str, str], dict] = {}
    per_image: dict[tuple[str, str], pd.DataFrame] = {}

    for results_dir in results_dirs:
        per_image_dir = Path(results_dir) / "per_image"
        if not per_image_dir.is_dir():
            print(f"  skip missing dir: {per_image_dir}")
            continue
        csv_paths = sorted({p for suf in CSV_SUFFIXES
                            for p in per_image_dir.glob(f"*{suf}.csv")})
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"  skip empty CSV: {csv_path.name}")
                continue
            # Prefer columns; fall back to filename parsing when absent.
            stem = csv_path.stem
            for suf in CSV_SUFFIXES:
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            fn_model, _, fn_dataset = stem.partition("_")
            model = str(df["model"].iloc[0]) if "model" in df.columns else fn_model
            dataset = str(df["dataset"].iloc[0]) if "dataset" in df.columns else fn_dataset
            key = (model, dataset)
            per_image[key] = df
            agg[key] = {k: df[k].mean() for k in metric_keys}
            agg[key]["num_images"] = len(df)

    if not agg:
        dirs = ", ".join(str(Path(d) / "per_image") for d in results_dirs)
        raise SystemExit(f"No per-image CSVs found in: {dirs}")
    return agg, per_image


def present(items, order):
    """Order ``items`` by ``order``, appending any extras alphabetically."""
    extras = sorted(set(items) - set(order))
    return [x for x in order if x in items] + extras


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def fig_grouped_bars(agg, per_image, out_dir):
    """2x2 panels, one metric each: datasets on x, bars grouped by model.

    Error bars are the std of the per-image metric (omitted if no CSV)."""
    datasets = present({d for _, d in agg}, DATASET_ORDER)
    models = present({m for m, _ in agg}, MODEL_ORDER)
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, label) in zip(axes.ravel(), METRICS):
        for i, model in enumerate(models):
            means, errs = [], []
            for dataset in datasets:
                a = agg.get((model, dataset))
                means.append(a[key] if a else np.nan)
                df = per_image.get((model, dataset))
                errs.append(df[key].std() if df is not None and key in df else 0.0)
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset, means, width, yerr=errs, capsize=3,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model), edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "alpha": 0.6},
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=11)
    fig.suptitle("Per-metric lane-detection comparison (error bars = per-image std)",
                 y=1.06, fontsize=13)
    fig.tight_layout()
    save(fig, out_dir, "fig1_grouped_bars")


def fig_boxplots(agg, per_image, out_dir, metric_key="lane_f1", metric_label="F1"):
    """Per-image distribution box plots: datasets on x, two boxes (models) each."""
    if not per_image:
        print("  skip fig2_boxplots: no per-image CSVs found")
        return
    datasets = present({d for _, d in agg}, DATASET_ORDER)
    models = present({m for m, _ in agg}, MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / max(len(models), 1)
    positions_center = np.arange(len(datasets))
    legend_handles = {}

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        data, pos = [], []
        for j, dataset in enumerate(datasets):
            df = per_image.get((model, dataset))
            if df is None or metric_key not in df:
                continue
            data.append(df[metric_key].to_numpy())
            pos.append(positions_center[j] + offset)
        if not data:
            continue
        color = MODEL_COLORS.get(model, "gray")
        bp = ax.boxplot(
            data, positions=pos, widths=width * 0.9, patch_artist=True,
            showfliers=False, medianprops={"color": "black"},
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.6, edgecolor="black", linewidth=0.6)
        # Overlay the mean as a diamond for a bar-chart-friendly reference point.
        means = [agg[(model, d)][metric_key] for d in datasets if (model, d) in agg]
        ax.scatter(pos, means, marker="D", color=color, edgecolor="black",
                   zorder=5, s=28)
        legend_handles[MODEL_LABELS.get(model, model)] = \
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6, ec="black")

    ax.set_xticks(positions_center)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=10)
    ax.set_ylabel(f"Per-image {metric_label}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"Per-image {metric_label} distribution by model and dataset\n"
                 "(box = IQR, line = median, ◆ = mean; outliers hidden)",
                 fontsize=12)
    ax.legend(legend_handles.values(), legend_handles.keys(), frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, out_dir, "fig2_boxplots")


def fig_radar(agg, out_dir):
    """Radar chart of each model's mean metric across its datasets."""
    models = present({m for m, _ in agg}, MODEL_ORDER)
    labels = [lbl for _, lbl in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for model in models:
        vals = []
        for key, _ in METRICS:
            xs = [v[key] for (m, _), v in agg.items() if m == model]
            vals.append(float(np.mean(xs)) if xs else np.nan)
        vals += vals[:1]
        color = MODEL_COLORS.get(model)
        ax.plot(angles, vals, color=color, linewidth=2,
                label=MODEL_LABELS.get(model, model))
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(180 / len(labels))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Mean metric profile per model\n(averaged across datasets)",
                 fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "fig3_radar")


def fig_precision_recall(agg, out_dir):
    """Scatter of precision vs recall, one marker per (model, dataset)."""
    datasets = present({d for _, d in agg}, DATASET_ORDER)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    ds_marker = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(8, 7))
    for (model, dataset), v in agg.items():
        ax.scatter(v["lane_recall"], v["lane_precision"],
                   color=MODEL_COLORS.get(model, "gray"),
                   marker=ds_marker[dataset], s=130, edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(DATASET_LABELS.get(dataset, dataset),
                    (v["lane_recall"], v["lane_precision"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Iso-F1 contour lines for reference.
    rr = np.linspace(0.01, 1, 200)
    for f1 in (0.3, 0.4, 0.5, 0.6, 0.7):
        pp = (f1 * rr) / (2 * rr - f1)
        mask = (pp > 0) & (pp <= 1)
        ax.plot(rr[mask], pp[mask], color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        # label near the right edge where the curve is in-bounds
        idx = np.where(mask)[0]
        if len(idx):
            ax.annotate(f"F1={f1}", (rr[idx[-1]], pp[idx[-1]]), fontsize=7,
                        color="gray", alpha=0.8)

    model_handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS[m],
                                markeredgecolor="black", markersize=10,
                                label=MODEL_LABELS.get(m, m))
                     for m in present({m for m, _ in agg}, MODEL_ORDER)]
    ds_handles = [plt.Line2D([], [], marker=ds_marker[d], linestyle="", color="gray",
                             markeredgecolor="black", markersize=9,
                             label=DATASET_LABELS.get(d, d))
                  for d in datasets]
    leg1 = ax.legend(handles=model_handles, title="Model", loc="lower left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, title="Dataset", loc="upper right", frameon=True)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Precision–Recall tradeoff (dotted = iso-F1)", fontsize=13)
    fig.tight_layout()
    save(fig, out_dir, "fig4_precision_recall")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["all"],
                    choices=MODEL_ORDER + ["all"], metavar="MODEL",
                    help="models to plot: any of %s, or 'all' (default)"
                         % ", ".join(MODEL_ORDER))
    ap.add_argument("--results-dir", nargs="+", type=Path,
                    default=DEFAULT_RESULTS_DIRS, metavar="DIR",
                    help="one or more result dirs; each must contain a per_image/ "
                         "subdir. Default searches the known dirs for all models.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/figures"),
                    help="where to write figures (default: outputs/figures)")
    args = ap.parse_args()

    agg, per_image = load_results(args.results_dir)
    available = sorted({m for m, _ in agg})

    # Filter to the requested models (keep order via MODEL_ORDER).
    selected = MODEL_ORDER if "all" in args.models else args.models
    agg = {(m, d): v for (m, d), v in agg.items() if m in selected}
    per_image = {(m, d): v for (m, d), v in per_image.items() if m in selected}
    if not agg:
        raise SystemExit(
            f"No data for models {selected}. Found models: {available or 'none'}.")

    found_models = present({m for m, _ in agg}, MODEL_ORDER)
    print(f"Loaded {len(agg)} runs for models: {', '.join(found_models)}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "savefig.facecolor": "white"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating figures:")
    fig_grouped_bars(agg, per_image, args.out_dir)
    fig_boxplots(agg, per_image, args.out_dir, metric_key="lane_f1", metric_label="F1")
    fig_radar(agg, args.out_dir)
    fig_precision_recall(agg, args.out_dir)
    print(f"Done. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
