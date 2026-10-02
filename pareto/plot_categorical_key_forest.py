"""
Compact 2-panel forest plot: the DIRECT effect on PR (net of MA) of the two strongest
categorical hyperparameter findings side by side -- head=CosFace (left) and
optimizer=SGD (right) -- for a single, paper-ready figure summarizing "the most
important categorical attributes" without needing the full set of individual
forest_head_*/forest_optimizer_* plots.

Reuses plot_hpo_analysis.py's own _forest_panel (unchanged) and reads directly from
the already-computed summary CSVs (pareto/hpo_analysis/summary/), exactly like
plot_hpo_analysis.py's own main() does -- no recomputation, so this is guaranteed
consistent with every other forest plot in this project.

Run locally:
    python3 pareto/plot_categorical_key_forest.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_hpo_analysis import _forest_panel, INK_PRIMARY, SURFACE  # noqa: E402

SUMMARY_DIR = os.path.join(HERE, "hpo_analysis", "summary")
PLOTS_DIR = os.path.join(HERE, "hpo_analysis", "plots")

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK_PRIMARY,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# (hyperparameter, term, panel title)
TARGETS = [
    ("head", "head[CosFace]", "Head = CosFace"),
    ("optimizer", "optimizer[SGD]", "Optimizer = SGD"),
]


def _pooled_row(summary: pd.DataFrame, hp: str, term: str) -> pd.Series:
    sel = summary[(summary.hyperparameter == hp) & (summary.term == term)
                 & (summary.effect_type == "direct_net_of_MA") & (summary.target == "PR")
                 & (summary.row_set == "final_budget_only") & (~summary.weighted)]
    return sel.iloc[0]


def _per_dir_rows(per_dir_df: pd.DataFrame, hp: str, term: str) -> pd.DataFrame:
    sel = (per_dir_df.hyperparameter == hp) & (per_dir_df.term == term) \
        & (per_dir_df.effect_type == "direct_net_of_MA") & (per_dir_df.target == "PR") \
        & (per_dir_df.row_set == "final_budget_only") & (~per_dir_df.weighted)
    return per_dir_df[sel]


def main():
    summary = pd.read_csv(os.path.join(SUMMARY_DIR, "hpo_effect_summary.csv"))
    per_dir_df = pd.read_csv(os.path.join(SUMMARY_DIR, "per_dir_effects.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (hp, term, panel_title) in zip(axes, TARGETS):
        pooled = _pooled_row(summary, hp, term)
        rows = _per_dir_rows(per_dir_df, hp, term)
        _forest_panel(ax, rows, pooled, panel_title)
        ax.set_xlabel("effect (ρ, -1..1)", fontsize=13)

    fig.suptitle("Direct effect on privacy risk (net of MA): key categorical hyper-parameters",
                fontsize=16, color=INK_PRIMARY, y=1.04)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "forest_categorical_key_net_of_MA.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
