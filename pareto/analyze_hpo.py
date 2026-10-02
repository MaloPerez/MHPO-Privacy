"""
Analyze hpo_dataset.csv (produced by baselines/collect_hpo_dataset.py) to find how
ParEGO/SMAC hyperparameters affect privacy risk (PR) and main-task accuracy (MA).

Effects are estimated per-dir first (the 16 source dirs have very different baseline
MA/PR ranges) and then combined across dirs via DerSimonian-Laird random-effects
meta-analysis, rather than pooling raw rows. Each hyperparameter gets both a TOTAL
effect on PR and a DIRECT effect controlling for MA, since the two are not
independent (privacy-utility tradeoff). Effect estimation uses per-dir OLS on
rank-transformed PR/MA, which recovers Spearman's rho (continuous hyperparameters) or
a rank-biserial effect (categorical) in one consistent, meta-analyzable form.

Near-total training collapse (MA below FAILURE_THRESHOLD) is modeled separately via
logistic regression and excluded from the MA/PR-magnitude analyses, since a collapsed
model's PR is noise rather than a real leakage signal.

Usage (no cluster/GPU needed)
------------------------------
python3 pareto/analyze_hpo.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "hpo_dataset.csv")
OUT_DIR = os.path.join(HERE, "hpo_analysis")
SUMMARY_DIR = os.path.join(OUT_DIR, "summary")

FAILURE_THRESHOLD = 0.05
MIN_ROWS_PER_GROUP = 8  # skip a per-dir fit if a group is too small to trust

CONTINUOUS_HPS = ["lr", "weight_decay", "batch_size"]
CATEGORICAL_HPS = {
    # hyperparameter -> reference level (chosen so the reference exists in every dir)
    "optimizer": "Adam",
    "loss": "Focal",
    # head's choice set differs by dir (see configs/config_multi.py): either
    # {Baseline, ArcFace, CosFace} or {CosFace, ArcFace, Linear}. 'ArcFace' is the
    # only level common to both, so it's the only sensible global reference.
    "head": "ArcFace",
}


# ---------------------------------------------------------------------------
# Meta-analysis (DerSimonian-Laird random effects)
# ---------------------------------------------------------------------------

def _meta_dl(y: np.ndarray, v: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects meta-analysis of point estimates y with
    known sampling variances v. Returns pooled estimate, SE, two-sided p, tau^2, k."""
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    v = np.clip(v, 1e-8, None)
    w_fixed = 1.0 / v
    y_bar_fixed = np.sum(w_fixed * y) / np.sum(w_fixed)
    k = len(y)
    if k < 2:
        se = float(np.sqrt(v[0])) if k == 1 else np.nan
        return dict(estimate=float(y[0]) if k == 1 else np.nan, se=se,
                    p=np.nan, tau2=0.0, k=k)
    Q = np.sum(w_fixed * (y - y_bar_fixed) ** 2)
    df = k - 1
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    w_random = 1.0 / (v + tau2)
    y_bar = np.sum(w_random * y) / np.sum(w_random)
    se = float(np.sqrt(1.0 / np.sum(w_random)))
    z = y_bar / se if se > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return dict(estimate=float(y_bar), se=se, p=p, tau2=float(tau2), k=k)


def meta_analyze(per_dir: list[dict]) -> dict:
    """per_dir: list of {'estimate': float, 'se': float, 'n': int, 'source_dir': str}."""
    if not per_dir:
        return dict(estimate=np.nan, se=np.nan, p=np.nan, tau2=np.nan, k=0,
                    median=np.nan, iqr_low=np.nan, iqr_high=np.nan,
                    sign_consistency_frac=np.nan, n_dirs=0)
    estimates = np.array([d["estimate"] for d in per_dir])
    ses = np.array([d["se"] for d in per_dir])
    pooled = _meta_dl(estimates, ses ** 2)
    signs = np.sign(estimates)
    nonzero = signs[signs != 0]
    # Consistency with the POOLED (DerSimonian-Laird) estimate's sign, not the raw
    # median's -- these can differ (checked empirically: 4/88 groups in this dataset,
    # all null findings with |estimate|<0.02) since the median is unweighted while the
    # pooled estimate is inverse-variance-weighted. The pooled estimate is what's
    # actually reported as "the effect" and cited in the paper text, so consistency
    # should be measured against it.
    sign_consistency = float(np.mean(nonzero == np.sign(pooled["estimate"]))) if len(nonzero) else np.nan
    return dict(
        estimate=pooled["estimate"], se=pooled["se"], p=pooled["p"], tau2=pooled["tau2"],
        k=pooled["k"], median=float(np.median(estimates)),
        iqr_low=float(np.percentile(estimates, 25)), iqr_high=float(np.percentile(estimates, 75)),
        sign_consistency_frac=sign_consistency, n_dirs=len(per_dir),
    )


# ---------------------------------------------------------------------------
# Per-dir rank-OLS effect estimation (unifies Spearman / rank-biserial)
# ---------------------------------------------------------------------------

def _rank(s: pd.Series) -> pd.Series:
    return s.astype(float).rank()


def _term_matrix(g: pd.DataFrame, hp: str) -> pd.DataFrame | None:
    """Predictor matrix for one hyperparameter: rank() if continuous, dummy-coded
    (vs. its fixed reference level) if categorical. Returns None if degenerate
    (e.g. the hp doesn't vary in this group, or a category is unrepresented)."""
    if hp in CONTINUOUS_HPS:
        col = g[hp].astype(float)
        if col.nunique() < 3:
            return None
        return pd.DataFrame({hp: _rank(col).values}, index=g.index)
    ref = CATEGORICAL_HPS[hp]
    levels = [l for l in g[hp].unique() if l != ref]
    if not levels or ref not in set(g[hp]):
        return None
    dummies = pd.DataFrame({f"{hp}[{lvl}]": (g[hp] == lvl).astype(float).values
                            for lvl in levels}, index=g.index)
    # drop dummy columns representing <MIN_ROWS_PER_GROUP/2 rows -- too sparse to trust
    keep = [c for c in dummies.columns if dummies[c].sum() >= MIN_ROWS_PER_GROUP / 2]
    if not keep:
        return None
    return dummies[keep]


def per_dir_effect(df: pd.DataFrame, hp: str, target: str, control_ma: bool,
                   weight_col: str | None = None) -> dict[str, list[dict]]:
    """
    For each dir (and, for hp=='lr', each optimizer stratum), fit
      rank(target) ~ [rank(MA) if control_ma] + term(hp)
    Returns {term_name: [ {estimate, se, n, source_dir}, ... ]} -- one list per
    dummy/continuous term (head/optimizer/loss can yield multiple terms).
    """
    out: dict[str, list[dict]] = {}
    strata = df.groupby("source_dir")
    for dname, g in strata:
        sub_groups = [(dname, g)]
        if hp == "lr":
            sub_groups = [(f"{dname}|{opt}", gg) for opt, gg in g.groupby("optimizer")]
        for label, gg in sub_groups:
            if len(gg) < MIN_ROWS_PER_GROUP:
                continue
            X = _term_matrix(gg, hp)
            if X is None:
                continue
            y = _rank(gg[target])
            if control_ma:
                X = X.copy()
                X["MA_rank"] = _rank(gg["MA"]).values
            Xc = sm.add_constant(X, has_constant="add")
            w = gg[weight_col].astype(float).values if weight_col else None
            try:
                model = sm.WLS(y.values, Xc.values, weights=w) if w is not None else sm.OLS(y.values, Xc.values)
                fit = model.fit()
            except Exception:
                continue
            # Categorical dummy coefficients on rank(target) are in raw rank units.
            # For a pure two-group comparison (n_total == n_pair), r_rb = 2*coef/n_pair
            # is the standard rank-biserial correlation, bounded to exactly [-1,1]. But
            # ranks here are always over the FULL stratum (y = rank(gg[target]) includes
            # every row, not just the two compared groups), so when a 3rd+ level is also
            # present (e.g. head's third architecture-specific option) and occupies
            # "middle" rank territory, the two compared groups can be pushed to occupy
            # more of the extreme ranks than n_pair/2 on either side -- 2/n_pair then
            # overcorrects and can push |r_rb| toward 2. The general rescale that maps
            # the true maximum-separation case (one group takes the top n_level ranks,
            # the other the bottom n_reference ranks, of the FULL n_total-row ranking)
            # to exactly +-1 regardless of any 3rd-level rows in between is
            # 1/(n_total - n_pair/2) -- this reduces to the standard 2/n_pair exactly
            # when n_total == n_pair (no 3rd level). Computed per-term (not once per
            # stratum) since each dummy column has its own group size.
            for i, col in enumerate(Xc.columns):
                if col in ("const", "MA_rank"):
                    continue
                term = col if hp not in CONTINUOUS_HPS else hp
                if hp in CONTINUOUS_HPS:
                    rescale = 1.0
                else:
                    n_level = float(Xc[col].sum())
                    n_reference = float((gg[hp] == CATEGORICAL_HPS[hp]).sum())
                    n_pair = n_level + n_reference
                    denom = len(gg) - n_pair / 2.0
                    rescale = 1.0 / denom if denom > 0 else np.nan
                se = float(fit.bse[i]) * rescale
                if not np.isfinite(se) or se <= 0:
                    continue
                out.setdefault(term, []).append(dict(
                    estimate=float(fit.params[i]) * rescale, se=se, n=len(gg), source_dir=label))
    return out


# ---------------------------------------------------------------------------
# Step 0: failure-mode logistic regression, per dir
# ---------------------------------------------------------------------------

def failure_effect(df: pd.DataFrame, hp: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for dname, g in df.groupby("source_dir"):
        sub_groups = [(dname, g)]
        if hp == "lr":
            sub_groups = [(f"{dname}|{opt}", gg) for opt, gg in g.groupby("optimizer")]
        for label, gg in sub_groups:
            if len(gg) < MIN_ROWS_PER_GROUP or gg["failed"].nunique() < 2:
                continue
            if hp in CONTINUOUS_HPS:
                x = np.log(gg[hp].astype(float))
                if x.nunique() < 3:
                    continue
                X = pd.DataFrame({hp: x.values}, index=gg.index)
            else:
                X = _term_matrix(gg, hp)
                if X is None:
                    continue
            Xc = sm.add_constant(X, has_constant="add")
            y = gg["failed"].astype(int).values
            try:
                fit = sm.GLM(y, Xc.values, family=sm.families.Binomial()).fit()
            except Exception:
                continue
            for i, col in enumerate(Xc.columns):
                if col == "const":
                    continue
                term = col if hp not in CONTINUOUS_HPS else hp
                se = float(fit.bse[i])
                if not np.isfinite(se) or se <= 0:
                    continue
                out.setdefault(term, []).append(dict(
                    estimate=float(fit.params[i]), se=se, n=len(gg), source_dir=label))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def summarize(per_dir_by_term: dict[str, list[dict]], per_dir_log: list[dict],
             **meta_cols) -> list[dict]:
    """Meta-analyze each term's per-dir estimates into a summary row, and also log
    every individual per-dir estimate (with the same identifying meta_cols) into
    per_dir_log, for forest-plot rendering later."""
    rows = []
    for term, per_dir in per_dir_by_term.items():
        m = meta_analyze(per_dir)
        rows.append(dict(term=term, **m, **meta_cols))
        for d in per_dir:
            per_dir_log.append(dict(term=term, **d, **meta_cols))
    return rows


def run_analysis() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["failed"] = df["MA"] < FAILURE_THRESHOLD
    df["n_budget_rows"] = df.groupby(["source_dir", "config_hash"])["budget"].transform("count")
    df["ihs_weight"] = 1.0 / df["n_budget_rows"]

    all_rows: list[dict] = []
    per_dir_log: list[dict] = []

    # --- Step 0: failure mode (uses all rows, both failed/not, per row_set) ---
    for row_set, sub in [("all", df), ("final_budget_only", df[df.is_final_budget_for_config])]:
        for hp in CONTINUOUS_HPS + list(CATEGORICAL_HPS):
            per_dir_by_term = failure_effect(sub, hp)
            all_rows += summarize(per_dir_by_term, per_dir_log, hyperparameter=hp,
                                  effect_type="failure_logit", target="failed",
                                  row_set=row_set, weighted=False)

    # --- Steps 1-3: MA/PR magnitude analyses, non-failed rows only ---
    clean = df[~df["failed"]]
    row_sets = {
        "all": clean,
        "final_budget_only": clean[clean.is_final_budget_for_config],
    }
    for row_set, sub in row_sets.items():
        weight_variants = [("all" if row_set == "all" else row_set, False)]
        if row_set == "all":
            weight_variants.append(("all", True))
        for _, weighted in weight_variants:
            wcol = "ihs_weight" if weighted else None
            for hp in CONTINUOUS_HPS + list(CATEGORICAL_HPS):
                # total effect on PR
                pd_pr_total = per_dir_effect(sub, hp, "PR", control_ma=False, weight_col=wcol)
                all_rows += summarize(pd_pr_total, per_dir_log, hyperparameter=hp, effect_type="total",
                                      target="PR", row_set=row_set, weighted=weighted)
                # direct effect on PR, net of MA
                pd_pr_direct = per_dir_effect(sub, hp, "PR", control_ma=True, weight_col=wcol)
                all_rows += summarize(pd_pr_direct, per_dir_log, hyperparameter=hp,
                                      effect_type="direct_net_of_MA", target="PR",
                                      row_set=row_set, weighted=weighted)
                # total effect on MA (the "indirect via MA" leg)
                pd_ma_total = per_dir_effect(sub, hp, "MA", control_ma=False, weight_col=wcol)
                all_rows += summarize(pd_ma_total, per_dir_log, hyperparameter=hp, effect_type="total",
                                      target="MA", row_set=row_set, weighted=weighted)

    summary = pd.DataFrame(all_rows)
    summary = flag_pruning_sensitivity(summary)
    per_dir_df = pd.DataFrame(per_dir_log)
    return summary, per_dir_df, df


def flag_pruning_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """A (hyperparameter, term, effect_type, target) rule is 'pruning_sensitive' if the
    meta-analysis point estimate's sign disagrees between: all/unweighted vs
    all/weighted vs final_budget_only. Only meaningful for the magnitude analyses."""
    key_cols = ["hyperparameter", "term", "effect_type", "target"]
    magnitude = summary[summary.effect_type != "failure_logit"].copy()

    def _sensitive(group: pd.DataFrame) -> bool | float:
        vals = [v for v in group["estimate"] if pd.notna(v)]
        if len(vals) < 2:
            return np.nan
        signs = {np.sign(v) for v in vals if v != 0}
        return len(signs) > 1

    flags = magnitude.groupby(key_cols).apply(_sensitive, include_groups=False)
    flags = flags.rename("pruning_sensitive").reset_index()
    summary = summary.merge(flags, on=key_cols, how="left")
    return summary


# ---------------------------------------------------------------------------
# MixedLM cross-check (corroboration only, printed not merged into the CSV)
# ---------------------------------------------------------------------------

def mixedlm_crosscheck(df: pd.DataFrame, log_path: str) -> None:
    clean = df[(~df["failed"]) & df.is_final_budget_for_config].copy()
    lines = ["MixedLM cross-check (PR ~ MA + hyperparameter, random intercept per source_dir)",
            "final_budget_only, non-failed rows. Corroboration only -- see summary/ for the",
            "primary per-dir + meta-analysis results.", "=" * 70]
    checks = [
        ("weight_decay", "PR ~ MA + weight_decay"),
        ("batch_size", "PR ~ MA + batch_size"),
        ("optimizer", "PR ~ MA + C(optimizer, Treatment('Adam'))"),
        ("loss", "PR ~ MA + C(loss, Treatment('Focal'))"),
        # 'ArcFace' matches CATEGORICAL_HPS["head"]'s reference level used everywhere
        # else in the analysis -- the only head choice common to both architecture
        # families (Baseline: {Baseline,ArcFace,CosFace}; other: {CosFace,ArcFace,Linear}).
        ("head", "PR ~ MA + C(head, Treatment('ArcFace'))"),
    ]
    for name, formula in checks:
        try:
            fit = smf.mixedlm(formula, data=clean, groups=clean["source_dir"]).fit()
            lines.append(f"\n[{name}]\n{fit.summary().tables[1].to_string()}")
        except Exception as e:
            lines.append(f"\n[{name}] failed: {e}")
    # lr, stratified by optimizer (conditional hyperparameter)
    for opt in ["Adam", "SGD"]:
        sub = clean[clean.optimizer == opt]
        try:
            fit = smf.mixedlm("PR ~ MA + lr", data=sub, groups=sub["source_dir"]).fit()
            lines.append(f"\n[lr | {opt}]\n{fit.summary().tables[1].to_string()}")
        except Exception as e:
            lines.append(f"\n[lr | {opt}] failed: {e}")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"MixedLM cross-check written to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary, per_dir_df, df = run_analysis()

    out_path = os.path.join(SUMMARY_DIR, "hpo_effect_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Wrote {len(summary)} summary rows to {out_path}")

    per_dir_path = os.path.join(SUMMARY_DIR, "per_dir_effects.csv")
    per_dir_df.to_csv(per_dir_path, index=False)
    print(f"Wrote {len(per_dir_df)} per-dir estimate rows to {per_dir_path}")

    # Console highlights: the "general rules" table for the primary regime
    primary = summary[(summary.row_set == "final_budget_only") & (~summary.weighted)
                      & (summary.effect_type.isin(["total", "direct_net_of_MA"]))
                      & (summary.target == "PR")]
    primary = primary.sort_values("sign_consistency_frac", ascending=False)
    print("\n--- Top hyperparameter effects on PR by sign-consistency (final-budget-only) ---")
    cols = ["hyperparameter", "term", "effect_type", "median", "sign_consistency_frac",
           "n_dirs", "meta_p" if "meta_p" in primary.columns else "p", "pruning_sensitive"]
    cols = [c for c in cols if c in primary.columns]
    with pd.option_context("display.width", 160, "display.max_rows", 40):
        print(primary[cols].rename(columns={"p": "meta_p"}).to_string(index=False))

    mixedlm_crosscheck(df, os.path.join(SUMMARY_DIR, "mixedlm_crosscheck.txt"))


if __name__ == "__main__":
    main()
