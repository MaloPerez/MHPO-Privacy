"""
Plot privacy-utility Pareto curves for each defense from a pareto_sweep.py JSON output.

For each defense, scatters all (MA, PR) points (mean across repeats, with error bars)
and draws the Pareto front (non-dominated set) as a connected line. Axes follow the
paper's convention: X = main task accuracy (higher better), Y = privacy risk =
adversary_balanced_accuracy - chance_level (lower better).

Usage
-----
python baselines/plot_pareto_curves.py \\
  --results results/pareto_gender_race.json \\
  --output  figures/pareto_gender_race.pdf

Optional: overlay a single-point result (e.g. M-HPO ours) with --extra-points.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Pareto front computation
# ---------------------------------------------------------------------------

def pareto_front(ma_vals: list[float], pr_vals: list[float]) -> list[int]:
    """
    Return indices of non-dominated points.
    A point is non-dominated if no other point has both higher MA and lower PR.
    """
    points = list(zip(ma_vals, pr_vals))
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has >= MA and <= PR, with at least one strict
            if points[j][0] >= points[i][0] and points[j][1] <= points[i][1]:
                if points[j][0] > points[i][0] or points[j][1] < points[i][1]:
                    dominated[i] = True
                    break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Color / marker palette
# ---------------------------------------------------------------------------

DEFENSE_STYLE: dict[str, dict] = {
    # Noise-injection family
    'noisy':      {'color': '#1f77b4', 'marker': 'o', 'label': 'Noisy (Gaussian)'},
    'pca':        {'color': '#ff7f0e', 'marker': 's', 'label': 'PCA + Laplace'},
    's_pca':      {'color': '#9467bd', 'marker': 'D', 'label': 'S-PCA + Laplace'},
    # Learned-defense family
    'cloak':      {'color': '#d62728', 'marker': '^', 'label': 'Cloak'},
    'dl':         {'color': '#2ca02c', 'marker': 'v', 'label': 'Disentangled (DL)'},
    'arl':        {'color': '#8c564b', 'marker': 'P', 'label': 'ARL'},
    # Single-objective HPO baseline
    'hpo':        {'color': '#000000', 'marker': 'X', 'label': 'HPO (single-obj)', 'ms': 12},
    # Ours
    'm_hpo':      {'color': '#e377c2', 'marker': '*', 'label': 'M-HPO (Ours)', 'ms': 14},
    'm_hpo_pca':  {'color': '#17becf', 'marker': 'h', 'label': 'M-HPO + PCA (Ours)', 'ms': 13},
}
# Canonical legend order (fixed, semantic grouping — not JSON-insertion order)
_LEGEND_ORDER = list(DEFENSE_STYLE.keys())


_FALLBACK_COLORS = ['#17becf', '#bcbd22', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
_fallback_assigned: dict[str, dict] = {}

def _style(defense: str) -> dict:
    if defense in DEFENSE_STYLE:
        return DEFENSE_STYLE[defense]
    if defense not in _fallback_assigned:
        idx = len(_fallback_assigned) % len(_FALLBACK_COLORS)
        markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'h']
        _fallback_assigned[defense] = {
            'color': _FALLBACK_COLORS[idx],
            'marker': markers[idx % len(markers)],
            'label': defense,
        }
    return _fallback_assigned[defense]


# ---------------------------------------------------------------------------
# Grouping repeated runs (for confidence intervals)
# ---------------------------------------------------------------------------

def _group_by_defense_and_param(results: list[dict]) -> dict[str, dict]:
    """
    Group results by defense, then by param_value, collecting the MA/PR of every
    repeat (run_idx) of that point. Results without a 'run_idx' (legacy single-run
    files) are treated as a single repeat.
    """
    grouped: dict[str, dict] = {}
    for r in results:
        if r.get('MA') is None or r.get('PR') is None:
            continue  # skip errored runs
        d = r['defense']
        pv = r['param_value']
        by_param = grouped.setdefault(d, {})
        entry = by_param.setdefault(pv, {'MA': [], 'PR': [], 'param_name': r['param_name']})
        entry['MA'].append(r['MA'])
        entry['PR'].append(r['PR'])
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_pareto(results: list[dict], title: str = '', extra_points: list[dict] | None = None,
                output: str = 'pareto.pdf', show_grid: bool = True,
                pareto_only: bool = False, no_annotations: bool = False,
                show_errorbars: bool = True, xlim: tuple[float, float] | None = None,
                ylim: tuple[float, float] | None = None, legend_loc: str = 'best',
                better_pos: tuple[float, float, float, float] = (0.92, 0.08, 0.85, 0.15),
                no_better_arrow: bool = False):
    """
    results: list of dicts with keys defense, param_name, param_value, MA, PR,
             and optionally run_idx (repeats of the same point, for CIs)
    extra_points: list of dicts with keys defense, MA, PR (single-point baselines)
    """
    grouped = _group_by_defense_and_param(results)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    legend_entries: dict[str, Line2D] = {}
    for defense, param_groups in grouped.items():
        param_values = list(param_groups.keys())
        param_name = next(iter(param_groups.values()))['param_name']
        ma_mean, ma_std, pr_mean, pr_std = [], [], [], []
        for pv in param_values:
            m, ms_ = _mean_std(param_groups[pv]['MA'])
            p, ps_ = _mean_std(param_groups[pv]['PR'])
            ma_mean.append(m); ma_std.append(ms_)
            pr_mean.append(p); pr_std.append(ps_)
        ma = np.array(ma_mean); ma_err = np.array(ma_std)
        pr = np.array(pr_mean); pr_err = np.array(pr_std)

        style = _style(defense)
        ms = style.get('ms', 7)

        front_idx = pareto_front(ma.tolist(), pr.tolist())

        # Scatter all points (semi-transparent) with error bars, unless pareto_only
        if not pareto_only:
            ax.errorbar(ma, pr, xerr=ma_err if show_errorbars else None,
                        yerr=pr_err if show_errorbars else None, fmt=style['marker'],
                        color=style['color'], markersize=ms, alpha=0.5,
                        linestyle='none', capsize=3, elinewidth=1, zorder=3)

        if len(front_idx) >= 1:
            front_idx_sorted = sorted(front_idx, key=lambda i: ma[i])
            front_ma = ma[front_idx_sorted]
            front_pr = pr[front_idx_sorted]
            front_ma_err = ma_err[front_idx_sorted]
            front_pr_err = pr_err[front_idx_sorted]
            if len(front_idx) > 1:
                ax.plot(front_ma, front_pr, color=style['color'], linewidth=1.3,
                        linestyle='-', alpha=0.55, zorder=4)
            ax.errorbar(front_ma, front_pr,
                        xerr=front_ma_err if show_errorbars else None,
                        yerr=front_pr_err if show_errorbars else None,
                        fmt=style['marker'], color=style['color'], markersize=ms,
                        linestyle='none', capsize=3, elinewidth=1.2, zorder=5,
                        markeredgecolor='black', markeredgewidth=0.5)

        legend_entries[defense] = Line2D(
            [], [], color=style['color'], marker=style['marker'],
            markersize=min(ms, 10), markeredgecolor='black', markeredgewidth=0.5,
            linestyle='-', linewidth=1.3, alpha=0.9, label=style['label'],
        )

        # Annotate the best trade-off point on the Pareto front
        if len(front_idx) > 0 and not no_annotations:
            best = max(front_idx, key=lambda i: ma[i] - pr[i])
            ax.annotate(
                f"{param_name}={param_values[best]}",
                xy=(ma[best], pr[best]),
                fontsize=6, color=style['color'],
                xytext=(3, 3), textcoords='offset points',
            )

    # Overlay extra single-point results (e.g. M-HPO, HPO)
    if extra_points:
        for ep in extra_points:
            d = ep.get('defense', 'extra')
            style = _style(d)
            ms = style.get('ms', 10)
            ax.scatter([ep['MA']], [ep['PR']], color=style['color'], marker=style['marker'],
                       s=ms ** 2, zorder=6, edgecolors='black', linewidths=0.8,
                       label=style['label'])
            legend_entries[d] = Line2D(
                [], [], color=style['color'], marker=style['marker'],
                markersize=min(ms, 10), markeredgecolor='black', markeredgewidth=0.5,
                linestyle='none', label=style['label'],
            )

    # "Better" arrow annotation — white halo so it stays legible over data.
    # Position is overridable (--better-pos) since the bottom-right corner is often
    # exactly where the strongest methods' points cluster, colliding with a fixed spot;
    # --no-better-arrow drops it entirely for combos where no position works cleanly.
    if not no_better_arrow:
        bx, by, btx, bty = better_pos
        bbox = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75)
        ax.annotate(
            '', xy=(bx, by), xycoords='axes fraction',
            xytext=(btx, bty), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0, mutation_scale=10,
                            shrinkA=0, shrinkB=0),
            zorder=10,
        )
        ax.text(btx + 0.005, bty + 0.005, 'Better', transform=ax.transAxes, fontsize=8, color='black',
                zorder=10, bbox=bbox)

    ax.set_xlabel('Main Task Accuracy (↑)', fontsize=13)
    ax.set_ylabel('Privacy Risk (↓)', fontsize=13)
    if title:
        ax.set_title(title, fontsize=12)
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.4)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ordered_handles = [legend_entries[d] for d in _LEGEND_ORDER if d in legend_entries]
    ordered_handles += [h for d, h in legend_entries.items() if d not in _LEGEND_ORDER]
    # Default 'best' lets matplotlib pick whichever corner has the least data, but it
    # doesn't know about the "Better" arrow annotation (always bottom-right) and can
    # pick that same corner, colliding with it -- --legend-loc overrides per-combo
    # when that happens (e.g. AGE->gender under IR-SE-50).
    ax.legend(handles=ordered_handles, fontsize=10, loc=legend_loc,
              framealpha=0.9, ncol=1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Plot Pareto curves from pareto_sweep output')
    p.add_argument('--results', required=True, nargs='+',
                   help='JSON file(s) produced by pareto_sweep.py (can merge multiple)')
    p.add_argument('--output', default='pareto_curves.pdf',
                   help='Output figure path (.pdf or .png)')
    p.add_argument('--extra-points', default=None,
                   help='JSON file with single-point results to overlay, e.g. M-HPO. '
                        'Format: [{"defense": "m_hpo", "MA": 0.883, "PR": 0.020}, ...]')
    p.add_argument('--title', default='', help='Figure title')
    p.add_argument('--no-grid', action='store_true')
    p.add_argument('--pareto-only', action='store_true',
                   help='Only plot Pareto-optimal points, hide dominated ones')
    p.add_argument('--no-annotations', action='store_true',
                   help='Do not annotate the best trade-off point with its hyperparameter value')
    p.add_argument('--no-error-bars', action='store_true',
                   help='Hide the +/- 1 std error bars (still uses the mean across repeats for point position)')
    p.add_argument('--xlim', type=float, nargs=2, default=None, metavar=('XMIN', 'XMAX'),
                   help='X-axis (Main Task Accuracy) limits, e.g. --xlim 0.5 1.0')
    p.add_argument('--ylim', type=float, nargs=2, default=None, metavar=('YMIN', 'YMAX'),
                   help='Y-axis (Privacy Risk) limits, e.g. --ylim 0.0 0.3')
    p.add_argument('--legend-loc', nargs='+', default=['best'],
                   help="Legend location, space-separated (no quoting needed), e.g. "
                        "--legend-loc upper left. Override when the default 'best' "
                        "collides with the fixed bottom-right 'Better' arrow annotation.")
    p.add_argument('--better-pos', type=float, nargs=4, default=[0.92, 0.08, 0.85, 0.15],
                   metavar=('ARROW_X', 'ARROW_Y', 'TEXT_X', 'TEXT_Y'),
                   help="'Better' arrow position in axes-fraction coords (arrow tip xy, "
                        "then arrow tail / text xy). Default sits bottom-right; override "
                        "when a combo's own points cluster there instead, e.g. "
                        "--better-pos 0.45 0.85 0.38 0.92 to move it top-left-ish.")
    p.add_argument('--no-better-arrow', action='store_true',
                   help="Drop the 'Better' arrow annotation entirely, for combos where "
                        "no position avoids colliding with data or the legend.")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Merge results from multiple files
    all_results = []
    experiment_info = {}
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data.get('results', []))
        if not experiment_info:
            experiment_info = data.get('experiment', {})

    extra_points = None
    if args.extra_points:
        with open(args.extra_points) as f:
            extra_points = json.load(f)

    def _pretty_task(name: str) -> str:
        # "race" is FairFace's internal column/task-key name (used throughout the
        # codebase for data loading, class-weight lookups, etc.) -- display-only
        # rename to "Ethnicity" here, matching the paper's own terminology, without
        # touching the underlying key anywhere else.
        return "Ethnicity" if name.strip().lower() == "race" else name

    title = args.title or (
        f"Privacy-Utility Pareto: {_pretty_task(experiment_info.get('main_task',''))} → "
        f"{_pretty_task(experiment_info.get('sensitive_task',''))}"
    )

    plot_pareto(
        results=all_results,
        title=title,
        extra_points=extra_points,
        output=args.output,
        show_grid=not args.no_grid,
        pareto_only=args.pareto_only,
        show_errorbars=not args.no_error_bars,
        no_annotations=args.no_annotations,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        legend_loc=' '.join(args.legend_loc),
        better_pos=tuple(args.better_pos),
        no_better_arrow=args.no_better_arrow,
    )

    # Print summary table (mean +/- std across repeats, Pareto front only)
    grouped = _group_by_defense_and_param(all_results)

    print("\n--- Pareto Summary (mean +/- std across repeats) ---")
    print(f"{'Defense':<12} {'Param':>10} {'MA':>16} {'PR':>16} {'n':>3}")
    print("-" * 62)
    for defense, param_groups in sorted(grouped.items()):
        param_values = list(param_groups.keys())
        ma_stats = [_mean_std(param_groups[pv]['MA']) for pv in param_values]
        pr_stats = [_mean_std(param_groups[pv]['PR']) for pv in param_values]
        pts_ma = [m for m, _ in ma_stats]
        pts_pr = [m for m, _ in pr_stats]
        front_idx = pareto_front(pts_ma, pts_pr)
        for i in sorted(front_idx, key=lambda i: pts_ma[i]):
            ma_m, ma_s = ma_stats[i]
            pr_m, pr_s = pr_stats[i]
            n = len(param_groups[param_values[i]]['MA'])
            print(f"{defense:<12} {str(param_values[i]):>10} "
                  f"{ma_m:>7.4f} +/- {ma_s:<5.4f} {pr_m:>7.4f} +/- {pr_s:<5.4f} {n:>3}")
    print("-" * 62)
