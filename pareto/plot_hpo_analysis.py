"""
Render the plots for analyze_hpo.py's output: forest plots (per-dir effect estimates
+ pooled meta-analysis), a failure-rate-by-lr plot, and small-multiples scatter plots.

Palette/marks follow the dataviz skill's validated reference palette (references/palette.md):
  - diverging blue<->red encodes effect SIGN in forest plots (a "polarity" job)
  - status colors (good/critical) encode training-failure state in the scatter plots
  - categorical slots 1/2 (blue/orange) encode optimizer identity in the failure-rate plot
  - hairline gridlines, >=8px markers, 2px lines, legend for >=2 series, text in ink
    tokens (never the series color) -- static PDF figures for a paper, so the
    interactivity/dark-mode steps of the skill (built for HTML charts) don't apply.

Run locally (after analyze_hpo.py has produced its CSVs):
    python3 pareto/plot_hpo_analysis.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_DIR = os.path.join(HERE, "hpo_analysis", "summary")
PLOTS_DIR = os.path.join(HERE, "hpo_analysis", "plots")
CSV_PATH = os.path.join(HERE, "hpo_dataset.csv")
FAILURE_THRESHOLD = 0.05

# --- dataviz reference palette (references/palette.md) ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
DIVERGING_NEG = "#2a78d6"   # blue: effect decreases the target
DIVERGING_POS = "#e34948"   # red: effect increases the target
DIVERGING_MID = "#f0efec"
CAT_1 = "#2a78d6"           # optimizer=Adam
CAT_2 = "#eb6834"           # optimizer=SGD
STATUS_GOOD = "#0ca30c"     # not failed
STATUS_CRITICAL = "#d03b3b" # failed (MA < threshold)

# Hyperparameter variable name -> paper-quality display name (forest plot suptitles).
HP_PRETTY = {
    "lr": "Learning Rate",
    "weight_decay": "Weight Decay",
    "batch_size": "Batch Size",
    "optimizer": "Optimizer",
    "loss": "Loss Function",
    "head": "Head Architecture",
}

# source_dir -> "Architecture-Dataset-MainTask-PrivateTask", for display only (plot
# labels/titles). "-ADD"/"Baseline" runs are VGG16, "-REBUTTAL"/IR_SE_50 runs are
# ResNet. CelebA task names (Gray_Hair) are merged to one token (GrayHair) so the
# four dash-separated parts stay unambiguous.
NAME_MAP = {
    # Display values only -- source_dir KEYS stay "race" (the literal FairFace column/
    # directory name used everywhere for data loading); "Ethnicity" here is a display-
    # only rename matching the paper's own terminology, same convention already used
    # in baselines/plot_pareto_curves.py's _pretty_task().
    "hpo-resnet-gender-race-ADD": "VGG16-FairFace-Gender-Ethnicity",
    "hpo-resnet-gender-age-ADD": "VGG16-FairFace-Gender-Age",
    "hpo-resnet-age-gender-ADD": "VGG16-FairFace-Age-Gender",
    "hpo-resnet-age-race-ADD": "VGG16-FairFace-Age-Ethnicity",
    "hpo-resnet-race-gender-ADD": "VGG16-FairFace-Ethnicity-Gender",
    "hpo-resnet-race-age-ADD": "VGG16-FairFace-Ethnicity-Age",
    "hpo-resnet-gender-race-REBUTTAL": "ResNet-FairFace-Gender-Ethnicity",
    "hpo-resnet-gender-age-REBUTTAL": "ResNet-FairFace-Gender-Age",
    "hpo-resnet-age-gender-REBUTTAL": "ResNet-FairFace-Age-Gender",
    "hpo-resnet-age-race-REBUTTAL": "ResNet-FairFace-Age-Ethnicity",
    "hpo-resnet-race-gender-REBUTTAL": "ResNet-FairFace-Ethnicity-Gender",
    "hpo-resnet-race-age-REBUTTAL": "ResNet-FairFace-Ethnicity-Age",
    "hpo-baseline-celeba-eyeglasses-male": "VGG16-CelebA-Eyeglasses-Male",
    "hpo-baseline-celeba-grey-hair-male": "VGG16-CelebA-GrayHair-Male",
    "hpo-baseline-celeba-male-eyeglasses": "VGG16-CelebA-Male-Eyeglasses",
    "hpo-baseline-celeba-male-gray-hair": "VGG16-CelebA-Male-GrayHair",
}


def pretty_name(label: str) -> str:
    """Map a source_dir, optionally suffixed '|Adam'/'|SGD' (per-optimizer strata,
    see per_dir_effect), to its readable Architecture-Dataset-Main-Private form."""
    if "|" in label:
        dname, opt = label.split("|", 1)
        return f"{NAME_MAP.get(dname, dname)} ({opt})"
    return NAME_MAP.get(label, label)

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    png_path = os.path.splitext(path)[0] + ".png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} (+ .png)")


# ---------------------------------------------------------------------------
# Forest plots: per-dir estimates + pooled meta-analysis diamond
# ---------------------------------------------------------------------------

POOLED_BAND = "#dce8f7" # shaded pooled-CI band


def _forest_panel(ax, per_dir: pd.DataFrame, pooled_row: pd.Series, title: str):
    per_dir = per_dir.sort_values("estimate")
    y = np.arange(len(per_dir))
    ci = 1.96 * per_dir["se"].values
    est = per_dir["estimate"].values

    pest, pse = pooled_row["estimate"], pooled_row["se"]
    pci = 1.96 * pse if pd.notna(pse) else 0

    # shaded pooled 95% CI band, drawn first so it reads as context behind the dots
    ax.axvspan(pest - pci, pest + pci, color=POOLED_BAND, zorder=0, linewidth=0)
    ax.axvline(pest, color=DIVERGING_NEG if pest < 0 else DIVERGING_POS,
              linewidth=1, linestyle="--", alpha=0.55, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=1)

    colors = np.where(est > 0, DIVERGING_POS, DIVERGING_NEG)
    ax.errorbar(est, y, xerr=ci, fmt="none", ecolor=INK_MUTED,
               elinewidth=1, capsize=0, zorder=2, alpha=0.7)
    ax.scatter(est, y, c=colors, s=36, zorder=3, edgecolors=SURFACE, linewidths=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(per_dir["source_dir"].map(pretty_name), fontsize=11, color=INK_SECONDARY)
    ax.set_ylim(-1.5, len(per_dir) + 1.2)

    # pooled meta-analysis diamond, drawn above the per-dir rows
    py = len(per_dir) + 0.6
    ax.errorbar([pest], [py], xerr=[pci], fmt="none", ecolor=INK_PRIMARY, elinewidth=1.8, zorder=4)
    ax.scatter([pest], [py], marker="D", s=110, c=INK_PRIMARY, zorder=5,
              edgecolors=SURFACE, linewidths=1.2)
    ax.axhline(len(per_dir) - 0.4, color=GRID, linewidth=0.8)

    n = pooled_row.get("n_dirs", pooled_row.get("k", np.nan))
    sc = pooled_row.get("sign_consistency_frac", np.nan)
    p = pooled_row.get("p", np.nan)
    ax.set_title(title, fontsize=14, color=INK_PRIMARY, pad=28)
    ax.grid(axis="x", linewidth=0.8, alpha=0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    subtitle = f"pooled={pest:+.2f}  sign-consistency={sc:.0%} (n={int(n)})  p={p:.1g}"
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
               textcoords="offset points", fontsize=10.5, color=INK_SECONDARY,
               ha="left", va="bottom", linespacing=1.5)


def make_forest_plots(summary: pd.DataFrame, per_dir_df: pd.DataFrame):
    """One 2-panel (total | direct-net-of-MA) forest plot per hyperparameter/term,
    for the primary regime: row_set=final_budget_only, unweighted, target=PR."""
    sel = summary[(summary.row_set == "final_budget_only") & (~summary.weighted)
                 & (summary.target == "PR") & (summary.n_dirs >= 3)]
    for (hp, term), rows in sel.groupby(["hyperparameter", "term"]):
        total_row = rows[rows.effect_type == "total"]
        direct_row = rows[rows.effect_type == "direct_net_of_MA"]
        if total_row.empty:
            continue
        n_rows = int(total_row.iloc[0].get("n_dirs", 16))
        height = 4.8 * max(1.0, n_rows / 16)
        fig, axes = plt.subplots(1, 2 if not direct_row.empty else 1,
                                 figsize=(11 if not direct_row.empty else 5.5, height), sharey=True)
        axes = np.atleast_1d(axes)

        def _pd_for(effect_type):
            m = ((per_dir_df.row_set == "final_budget_only") & (~per_dir_df.weighted)
                & (per_dir_df.target == "PR") & (per_dir_df.hyperparameter == hp)
                & (per_dir_df.term == term) & (per_dir_df.effect_type == effect_type))
            return per_dir_df[m]

        _forest_panel(axes[0], _pd_for("total"), total_row.iloc[0], "Total effect on PR")
        if not direct_row.empty:
            _forest_panel(axes[1], _pd_for("direct_net_of_MA"), direct_row.iloc[0],
                         "Direct effect on PR (net of MA)")
        for ax in axes:
            ax.set_xlabel("effect (ρ, -1..1)", fontsize=13)
        fig.suptitle(HP_PRETTY.get(hp, hp) + (f" = {term.split('[')[-1].rstrip(']')}" if "[" in term else ""),
                    fontsize=16, color=INK_PRIMARY, y=1.04)
        safe_term = term.replace("[", "_").replace("]", "").replace("|", "_")
        _save(fig, os.path.join(PLOTS_DIR, f"forest_{hp}_{safe_term}.pdf"))


# ---------------------------------------------------------------------------
# Failure-rate vs lr, split by optimizer
# ---------------------------------------------------------------------------

def make_failure_rate_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    for opt, color in [("Adam", CAT_1), ("SGD", CAT_2)]:
        sub = df[df.optimizer == opt].copy()
        sub["log_lr"] = np.log10(sub["lr"])
        bins = pd.qcut(sub["log_lr"], 6, duplicates="drop")
        agg = sub.groupby(bins, observed=True).agg(
            rate=("failed", "mean"), center=("lr", "median"), n=("failed", "size"))
        agg = agg.sort_values("center")
        ax.plot(agg["center"], agg["rate"], color=color, linewidth=2, marker="o",
               markersize=8, markerfacecolor=color, markeredgecolor=SURFACE,
               markeredgewidth=1.2, label=opt, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("fraction of configs with MA < 0.05 (training failure)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Training-failure rate rises with learning rate", fontsize=12, color=INK_PRIMARY)
    ax.grid(axis="y", linewidth=0.8, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", title="optimizer", title_fontsize=9)
    _save(fig, os.path.join(PLOTS_DIR, "failure_rate_vs_lr.pdf"))


# ---------------------------------------------------------------------------
# Small multiples: hyperparameter vs PR, one panel per source_dir, colored by failed
# ---------------------------------------------------------------------------

TREND_LINE = "#222222"
OPT_TREND_COLORS = {"Adam": CAT_1, "SGD": CAT_2}


def _add_trend(ax, x, y, color=TREND_LINE, log_x=True, min_per_bin=12, **line_kw):
    """Bold binned-median trend line, so the slope reads through per-point scatter.
    Bin count adapts to sample size (>= min_per_bin points/bin) -- a fixed bin count
    made small panels (as few as ~20 points) zigzag on per-bin sampling noise rather
    than show a readable trend."""
    n_bins = max(3, min(6, len(x) // min_per_bin))
    if len(x) < 2 * min_per_bin:
        return
    t = pd.DataFrame({"x": x, "y": y})
    t["kx"] = np.log10(t["x"]) if log_x else t["x"]
    bins = pd.qcut(t["kx"], min(n_bins, t["kx"].nunique()), duplicates="drop")
    agg = t.groupby(bins, observed=True).agg(x=("x", "median"), y=("y", "median")).sort_values("x")
    ax.plot(agg["x"], agg["y"], color=color, linewidth=2, marker="o", markersize=4,
           zorder=4, **line_kw)


BUDGET_SIZES = {13: 20, 40: 55, 120: 110}


def make_small_multiples(df: pd.DataFrame, hp: str, log_x: bool, optimizer: str | None = None):
    """One 16-panel grid; point size = training budget (epochs), color = main task
    accuracy. lr is plotted separately per optimizer since its search range differs
    between Adam and SGD."""
    dirs = sorted(df.source_dir.unique())
    ncols, nrows = 4, int(np.ceil(len(dirs) / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.atleast_2d(axes)
    sm = None
    for i, dname in enumerate(dirs):
        ax = axes[i // ncols, i % ncols]
        sub = df[(df.source_dir == dname) & (df.is_final_budget_for_config)]
        if optimizer is not None:
            sub = sub[sub.optimizer == optimizer]
        sizes = sub["budget"].map(BUDGET_SIZES).fillna(30)
        sm = ax.scatter(sub[hp], sub["PR"], c=sub["MA"], s=sizes, cmap="RdYlBu_r",
                        vmin=0, vmax=1, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3)
        _add_trend(ax, sub.loc[~sub.failed, hp].values, sub.loc[~sub.failed, "PR"].values, log_x=log_x)
        if log_x:
            ax.set_xscale("log")
        ax.set_title(pretty_name(dname), fontsize=7, color=INK_SECONDARY)
        ax.grid(linewidth=0.6, alpha=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=6.5)
    for j in range(len(dirs), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    size_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="grey",
                           markersize=np.sqrt(s), label=f"{b} epochs")
                    for b, s in BUDGET_SIZES.items()]
    size_handles.append(Line2D([0], [0], color=TREND_LINE, linewidth=2, marker="o",
                               markersize=4, label="median trend (trained only)"))
    fig.legend(handles=size_handles, loc="upper center", ncol=len(size_handles), frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, 1.05), title="Training budget", title_fontsize=9)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("Main Task Accuracy", fontsize=9)

    opt_suffix = f" ({optimizer} only)" if optimizer else ""
    fig.suptitle(f"{hp} vs privacy risk (PR), all 16 experiments{opt_suffix}", fontsize=13,
                color=INK_PRIMARY, y=1.09)
    fig.supxlabel(hp, fontsize=10, color=INK_SECONDARY)
    fig.supylabel("PR (chance-adjusted)", fontsize=10, color=INK_SECONDARY)
    fname_suffix = f"_{optimizer}" if optimizer else ""
    _save(fig, os.path.join(PLOTS_DIR, f"small_multiples_{hp}_vs_PR{fname_suffix}.pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summary = pd.read_csv(os.path.join(SUMMARY_DIR, "hpo_effect_summary.csv"))
    per_dir_df = pd.read_csv(os.path.join(SUMMARY_DIR, "per_dir_effects.csv"))
    df = pd.read_csv(CSV_PATH)
    df["failed"] = df["MA"] < FAILURE_THRESHOLD

    print("Forest plots...")
    make_forest_plots(summary, per_dir_df)

    print("Failure-rate plot...")
    make_failure_rate_plot(df)

    print("Small-multiples scatter plots...")
    make_small_multiples(df, "weight_decay", log_x=True)
    make_small_multiples(df, "lr", log_x=True, optimizer="Adam")
    make_small_multiples(df, "lr", log_x=True, optimizer="SGD")

    print(f"\nAll plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
