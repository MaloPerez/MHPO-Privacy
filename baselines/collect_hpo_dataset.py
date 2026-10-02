"""
Walk all multi-objective HPO run directories (MultiObj/ParEGOTain.py output) and
consolidate every trained checkpoint's hyperparameters, main-task accuracy (MA), and
privacy risk (PR) into a single CSV for downstream analysis (pareto/analyze_hpo.py).

Every checkpoint already stores everything needed, so this is a pure read-only,
CPU-only scan (no GPU, no re-evaluation, no dataset loading). Unlike
discover_best_models.py / parego_to_sweep_json.py, it keeps EVERY budget checkpoint
for EVERY config (not just the highest), tagging each row so the analysis phase can
filter later (is_final_budget_for_config, reached_experiment_max_budget).

Usage
-----
python baselines/collect_hpo_dataset.py --output hpo_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import warnings

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_ROOT = os.path.join(_ROOT, "multiHPO_results")

NUM_CLASS_DICT = {"race": 7, "age": 9, "gender": 2}

HYPERPARAM_FIELDS = ["head", "loss", "optimizer", "batch_size", "weight_decay", "lr"]

CSV_FIELDS = [
    "source_dir", "architecture", "dataset", "main_task", "sensitive_task", "embedding_size",
    "config_hash", "budget", "is_final_budget_for_config", "reached_experiment_max_budget",
    "head", "loss", "optimizer", "batch_size", "weight_decay", "lr",
    "MA", "PR_raw", "PR",
]


def num_class_for(sensitive_task: str, dataset: str) -> int:
    if dataset == "CelebA":
        return 2
    return NUM_CLASS_DICT[sensitive_task.lower()]


# ---------------------------------------------------------------------------
# Experiment registry -- directory name -> main_task/sensitive_task/dataset/backbone.
# The 10 entries marked "from discover_best_models.py" are copied verbatim from that
# script's MULTI_EXPERIMENTS registry (exact-match dir names). The 6 "-ADD" entries are
# added here: same main/sensitive-task pairing as their "-REBUTTAL" counterparts, but
# Baseline architecture (embedding_size=4608) -- the "resnet" in their directory
# names is a legacy naming mistake, not the actual architecture used.
# ---------------------------------------------------------------------------

MULTI_EXPERIMENTS: dict[str, dict] = {
    # --- Baseline / FairFace ("-ADD") ---
    "hpo-resnet-gender-race-ADD": dict(main_task="GENDER", sensitive_task="race",
                                       dataset="FairFace", backbone="Baseline", embedding_size=4608),
    "hpo-resnet-gender-age-ADD": dict(main_task="GENDER", sensitive_task="age",
                                      dataset="FairFace", backbone="Baseline", embedding_size=4608),
    "hpo-resnet-age-gender-ADD": dict(main_task="AGE", sensitive_task="gender",
                                      dataset="FairFace", backbone="Baseline", embedding_size=4608),
    "hpo-resnet-age-race-ADD": dict(main_task="AGE", sensitive_task="race",
                                    dataset="FairFace", backbone="Baseline", embedding_size=4608),
    "hpo-resnet-race-gender-ADD": dict(main_task="RACE", sensitive_task="gender",
                                       dataset="FairFace", backbone="Baseline", embedding_size=4608),
    "hpo-resnet-race-age-ADD": dict(main_task="RACE", sensitive_task="age",
                                    dataset="FairFace", backbone="Baseline", embedding_size=4608),

    # --- IR_SE_50 / FairFace ("-REBUTTAL"), from discover_best_models.py ---
    "hpo-resnet-gender-race-REBUTTAL": dict(main_task="GENDER", sensitive_task="race",
                                            dataset="FairFace", backbone="IR_SE_50", embedding_size=512),
    "hpo-resnet-gender-age-REBUTTAL": dict(main_task="GENDER", sensitive_task="age",
                                           dataset="FairFace", backbone="IR_SE_50", embedding_size=512),
    "hpo-resnet-race-gender-REBUTTAL": dict(main_task="RACE", sensitive_task="gender",
                                            dataset="FairFace", backbone="IR_SE_50", embedding_size=512),
    "hpo-resnet-race-age-REBUTTAL": dict(main_task="RACE", sensitive_task="age",
                                         dataset="FairFace", backbone="IR_SE_50", embedding_size=512),
    "hpo-resnet-age-gender-REBUTTAL": dict(main_task="AGE", sensitive_task="gender",
                                           dataset="FairFace", backbone="IR_SE_50", embedding_size=512),
    "hpo-resnet-age-race-REBUTTAL": dict(main_task="AGE", sensitive_task="race",
                                         dataset="FairFace", backbone="IR_SE_50", embedding_size=512),

    # --- Baseline / CelebA, from discover_best_models.py ---
    "hpo-baseline-celeba-eyeglasses-male": dict(main_task="Eyeglasses", sensitive_task="Male",
                                                dataset="CelebA", backbone="Baseline", embedding_size=4608),
    "hpo-baseline-celeba-grey-hair-male": dict(main_task="Gray_Hair", sensitive_task="Male",
                                               dataset="CelebA", backbone="Baseline", embedding_size=4608),
    "hpo-baseline-celeba-male-eyeglasses": dict(main_task="Male", sensitive_task="Eyeglasses",
                                                dataset="CelebA", backbone="Baseline", embedding_size=4608),
    "hpo-baseline-celeba-male-gray-hair": dict(main_task="Male", sensitive_task="Gray_Hair",
                                               dataset="CelebA", backbone="Baseline", embedding_size=4608),

    # --- Single-hyperparameter validation sweeps (MultiObj/hyperparam_sweep.py), combos never
    # trained elsewhere in this project. Same checkpoint format, so no code changes
    # needed here -- just registry entries. Each is really two runs (a weight_decay
    # sweep and an lr sweep) sharing one exp_path/model_checkpoints dir, since
    # hyperparam_sweep.py's config_hash already keeps every swept value distinct. ---
    "validation-celeba-male-wearinghat": dict(main_task="Male", sensitive_task="Wearing_Hat",
                                              dataset="CelebA", backbone="Baseline", embedding_size=4608),
    "validation-resnet-celeba-male-eyeglasses": dict(main_task="Male", sensitive_task="Eyeglasses",
                                                     dataset="CelebA", backbone="IR_SE_50", embedding_size=512),
}


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def find_checkpoint_dir(exp_name: str) -> str | None:
    d = os.path.join(MULTI_ROOT, exp_name, "model_checkpoints")
    return d if os.path.isdir(d) else None


def scan_all_budgets(model_checkpoints_dir: str) -> list[dict]:
    """
    For every config-hash subdirectory, load EVERY budget_*_checkpoint.pth (map_location='cpu'),
    and return one raw record per (config_hash, budget). Ground truth is the checkpoint files
    themselves, matching discover_best_models.py's/parego_to_sweep_json.py's approach.
    """
    out = []
    for hash_dir in sorted(glob.glob(os.path.join(model_checkpoints_dir, "*"))):
        if not os.path.isdir(hash_dir):
            continue
        config_hash = os.path.basename(hash_dir)
        budget_files = []
        for fname in os.listdir(hash_dir):
            m = re.match(r"budget_(\d+)_checkpoint\.pth", fname)
            if m:
                budget_files.append((int(m.group(1)), os.path.join(hash_dir, fname)))
        if not budget_files:
            continue
        max_budget_for_config = max(b for b, _ in budget_files)
        for budget, ckpt_path in sorted(budget_files):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                warnings.warn(f"Failed to load {ckpt_path}: {e}")
                continue
            out.append(dict(
                config_hash=config_hash,
                budget=budget,
                is_final_budget_for_config=(budget == max_budget_for_config),
                ckpt=ckpt,
            ))
    return out


def build_row(entry: dict, exp_name: str, cfg: dict, experiment_max_budget: int) -> dict:
    ckpt = entry["ckpt"]
    num_class = num_class_for(cfg["sensitive_task"], cfg["dataset"])
    sens_key = f"privacy_risk_{cfg['sensitive_task']}"

    one_minus_ma = ckpt.get("1 - main_task_accuracy")
    ma = (1.0 - float(one_minus_ma)) if one_minus_ma is not None else None
    if one_minus_ma is None:
        warnings.warn(f"[{exp_name}/{entry['config_hash']}/budget_{entry['budget']}] "
                      f"missing '1 - main_task_accuracy'")

    pr_raw = ckpt.get(sens_key)
    if pr_raw is None:
        for k, v in ckpt.items():
            if isinstance(k, str) and k.lower() == sens_key.lower():
                pr_raw = v
                break
    if pr_raw is None:
        warnings.warn(f"[{exp_name}/{entry['config_hash']}/budget_{entry['budget']}] "
                      f"missing '{sens_key}'")
    pr = max(0.0, float(pr_raw) - 1.0 / num_class) if pr_raw is not None else None

    row = dict(
        source_dir=exp_name,
        architecture=cfg["backbone"],
        dataset=cfg["dataset"],
        main_task=cfg["main_task"],
        sensitive_task=cfg["sensitive_task"],
        embedding_size=cfg["embedding_size"],
        config_hash=entry["config_hash"],
        budget=entry["budget"],
        is_final_budget_for_config=entry["is_final_budget_for_config"],
        reached_experiment_max_budget=(entry["budget"] == experiment_max_budget),
        MA=ma,
        PR_raw=float(pr_raw) if pr_raw is not None else None,
        PR=pr,
    )
    for field in HYPERPARAM_FIELDS:
        value = ckpt.get(field)
        if value is None:
            warnings.warn(f"[{exp_name}/{entry['config_hash']}/budget_{entry['budget']}] "
                          f"missing '{field}'")
        row[field] = value
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Consolidate ParEGO/SMAC HPO checkpoints (hyperparameters + MA + PR) "
                    "across all 16 experiment directories into one CSV.")
    p.add_argument("--output", default="hpo_dataset.csv")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Optional subset of experiment directory names to scan "
                        "(default: all 16 registered).")
    return p.parse_args()


def main():
    args = parse_args()
    exp_names = args.experiments if args.experiments else list(MULTI_EXPERIMENTS.keys())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    f = open(args.output, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()

    total_rows = 0
    total_configs = 0
    for exp_name in exp_names:
        print("=" * 70)
        print(f"[{exp_name}]")
        if exp_name not in MULTI_EXPERIMENTS:
            print(f"  SKIPPED: not in registry (unknown experiment name)")
            continue
        cfg = MULTI_EXPERIMENTS[exp_name]

        ckdir = find_checkpoint_dir(exp_name)
        if ckdir is None:
            print(f"  NOT FOUND: {os.path.join(MULTI_ROOT, exp_name, 'model_checkpoints')}")
            continue

        entries = scan_all_budgets(ckdir)
        if not entries:
            print("  no checkpoints found")
            continue

        experiment_max_budget = max(e["budget"] for e in entries)
        n_configs = len({e["config_hash"] for e in entries})

        rows = [build_row(e, exp_name, cfg, experiment_max_budget) for e in entries]
        for row in rows:
            writer.writerow(row)
        f.flush()

        n_final = sum(1 for r in rows if r["is_final_budget_for_config"])
        n_full = sum(1 for r in rows if r["reached_experiment_max_budget"])
        print(f"  dataset={cfg['dataset']}  main_task={cfg['main_task']}  "
             f"sensitive_task={cfg['sensitive_task']}  backbone={cfg['backbone']}")
        print(f"  {n_configs} configs, {len(rows)} checkpoint rows "
             f"({n_final} final-per-config, {n_full} reached experiment max budget={experiment_max_budget})")

        total_rows += len(rows)
        total_configs += n_configs

    f.close()
    print("=" * 70)
    print(f"Done. {total_configs} configs, {total_rows} checkpoint rows written to {args.output}")


if __name__ == "__main__":
    main()
