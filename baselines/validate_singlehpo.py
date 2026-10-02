"""
Cheap replication check for the hyperparameter -> privacy-risk rules found by the
multi-objective HPO analysis (pareto/analyze_hpo.py).

singleHPO_results/ (MultiObj/GeneralHPO.py) holds 6 single-objective HPO searches
(main-task accuracy only, privacy never an objective) over the same search space and
architectures/datasets as the multi-objective runs in hpo_dataset.csv. This script
evaluates privacy risk post-hoc on those already-trained checkpoints -- no new
backbone training, just a fresh adversary per checkpoint via pareto_sweep.py's own
_eval_privacy, reusing each config's multi-objective-HPO counterpart's adv_config so
the adversary methodology is directly comparable, not just similar.

Output uses the same CSV schema as pareto/hpo_dataset.csv, with
is_independent_replication=True, so analyze_hpo.py can consume it unmodified.

Usage (needs a GPU for the adversary training, unlike collect_hpo_dataset.py):
    python3 baselines/validate_singlehpo.py --output validation_singlehpo.csv

By default every config's own final checkpoint is evaluated regardless of what budget
it topped out at; add --max-budget-only to skip configs pruned before this
experiment's max observed budget.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import warnings

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

MULTI_ROOT = f"{_ROOT}/multiHPO_results"
SINGLE_ROOT = f"{_ROOT}/singleHPO_results"

HYPERPARAM_FIELDS = ["head", "loss", "optimizer", "batch_size", "weight_decay", "lr"]
CSV_FIELDS = [
    "source_dir", "architecture", "dataset", "main_task", "sensitive_task", "embedding_size",
    "config_hash", "budget", "is_final_budget_for_config", "reached_experiment_max_budget",
    "is_independent_replication",
    "head", "loss", "optimizer", "batch_size", "weight_decay", "lr",
    "MA", "PR",
]

# ---------------------------------------------------------------------------
# Registry: singleHPO dir -> metadata + default sensitive task + the counterpart
# multiHPO dir whose adv_config.pkl we reuse (same adversary methodology as the
# combo we're replicating against -- not just "similar").
# ---------------------------------------------------------------------------

SINGLE_EXPERIMENTS: dict[str, dict] = {
    "single-hpo-resnet-gender": dict(
        main_task="GENDER", dataset="FairFace", backbone="IR_SE_50", embedding_size=512,
        sensitive_task="race", counterpart="hpo-resnet-gender-race-REBUTTAL"),
    "single-hpo-resnet-race": dict(
        main_task="RACE", dataset="FairFace", backbone="IR_SE_50", embedding_size=512,
        sensitive_task="gender", counterpart="hpo-resnet-race-gender-REBUTTAL"),
    "single-hpo-resnet-age": dict(
        main_task="AGE", dataset="FairFace", backbone="IR_SE_50", embedding_size=512,
        sensitive_task="gender", counterpart="hpo-resnet-age-gender-REBUTTAL"),
    "single-hpo-celeba-male": dict(
        main_task="Male", dataset="CelebA", backbone="Baseline", embedding_size=4608,
        sensitive_task="Eyeglasses", counterpart="hpo-baseline-celeba-male-eyeglasses"),
    "single-hpo-celeba-eyeglasses": dict(
        main_task="Eyeglasses", dataset="CelebA", backbone="Baseline", embedding_size=4608,
        sensitive_task="Male", counterpart="hpo-baseline-celeba-eyeglasses-male"),
    "single-hpo-celeba-grayhair": dict(
        main_task="Gray_Hair", dataset="CelebA", backbone="Baseline", embedding_size=4608,
        sensitive_task="Male", counterpart="hpo-baseline-celeba-grey-hair-male"),
}


def _adv_config_path(counterpart: str) -> str:
    return os.path.join(MULTI_ROOT, counterpart, "adv_config", "adv_config.pkl")


def get_backbone_cls(name: str):
    from model_resnet import ResNet_50, ResNet_101, ResNet_152
    from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
    from baselines import Encoder
    return {
        'ResNet_50': ResNet_50, 'ResNet_101': ResNet_101, 'ResNet_152': ResNet_152,
        'IR_50': IR_50, 'IR_101': IR_101, 'IR_152': IR_152,
        'IR_SE_50': IR_SE_50, 'IR_SE_101': IR_SE_101, 'IR_SE_152': IR_SE_152,
        'Baseline': lambda s, e: Encoder(),
    }[name]


# ---------------------------------------------------------------------------
# Checkpoint scanning -- one (final-budget) checkpoint per config, same as
# discover_best_models.py's scan_hashes / collect_hpo_dataset.py's registry
# pattern, restricted to the final checkpoint only (not every budget) to keep
# this "cheap" -- 3x fewer adversary-training passes than collect_hpo_dataset.py's full-budget scan.
# ---------------------------------------------------------------------------

def find_checkpoint_dir(exp_name: str) -> str | None:
    d = os.path.join(SINGLE_ROOT, exp_name, "model_checkpoints")
    return d if os.path.isdir(d) else None


def scan_final_checkpoints(model_checkpoints_dir: str) -> list[dict]:
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
        budget, ckpt_path = max(budget_files, key=lambda t: t[0])
        out.append(dict(config_hash=config_hash, budget=budget, ckpt_path=ckpt_path))
    return out


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment(exp_name: str, cfg: dict, device: torch.device, adv_epochs: int,
                        write_row, done_keys: set[tuple[str, str]],
                        max_budget_only: bool = False,
                        input_size=(112, 112)) -> tuple[int, int, int]:
    """Evaluates each checkpoint, writing results incrementally so a crash doesn't lose
    completed work. Checkpoints already in done_keys or that fail are skipped, not
    retried. Returns (n_done, n_skipped, n_failed)."""
    from pareto_sweep import _load_datasets, _resolve_tasks, _eval_privacy, _get_adv_config

    sensitive_task = cfg["sensitive_task"]
    source_dir = f"{exp_name}->{sensitive_task}"
    full_cfg = dict(
        dataset=cfg["dataset"], main_task=cfg["main_task"], sensitive_task=sensitive_task,
        embedding_size=cfg["embedding_size"], input_size=list(input_size),
        adv_epochs=adv_epochs, adv_config_path=_adv_config_path(cfg["counterpart"]),
        batch_size=128,
    )

    ckdir = find_checkpoint_dir(exp_name)
    if ckdir is None:
        print(f"  NOT FOUND: {os.path.join(SINGLE_ROOT, exp_name, 'model_checkpoints')}")
        return (0, 0, 0)

    entries = scan_final_checkpoints(ckdir)
    if not entries:
        print("  no checkpoints found")
        return (0, 0, 0)
    experiment_max_budget = max(e["budget"] for e in entries)

    if max_budget_only:
        n_before = len(entries)
        entries = [e for e in entries if e["budget"] == experiment_max_budget]
        print(f"  --max-budget-only: {n_before - len(entries)} pruned-budget config(s) "
             f"skipped, {len(entries)} at budget={experiment_max_budget} remain")

    train_ds, val_ds, index_label, _ = _load_datasets(full_cfg)
    _, sens_key = _resolve_tasks(full_cfg, index_label)
    adv_cfg = _get_adv_config(full_cfg)
    backbone_cls = get_backbone_cls(cfg["backbone"])

    n_done = n_skipped = n_failed = 0
    for i, e in enumerate(entries):
        if (source_dir, e["config_hash"]) in done_keys:
            n_skipped += 1
            continue

        backbone = None
        try:
            ckpt = torch.load(e["ckpt_path"], map_location=device, weights_only=False)

            one_minus_ma = ckpt.get("1 - main_task_accuracy")
            if one_minus_ma is None:
                raise ValueError("missing '1 - main_task_accuracy'")
            ma = 1.0 - float(one_minus_ma)

            backbone = backbone_cls(full_cfg["input_size"], cfg["embedding_size"]).to(device)
            backbone.load_state_dict(ckpt["backbone_state_dict"])
            backbone.eval()

            pr = _eval_privacy(backbone, train_ds, val_ds, sens_key, full_cfg, adv_cfg, device)

            row = dict(
                source_dir=source_dir,
                architecture=cfg["backbone"], dataset=cfg["dataset"],
                main_task=cfg["main_task"], sensitive_task=sensitive_task,
                embedding_size=cfg["embedding_size"], config_hash=e["config_hash"],
                budget=e["budget"], is_final_budget_for_config=True,
                reached_experiment_max_budget=(e["budget"] == experiment_max_budget),
                is_independent_replication=True,
                MA=ma, PR=pr,
            )
            for field in HYPERPARAM_FIELDS:
                value = ckpt.get(field)
                if value is None:
                    warnings.warn(f"[{exp_name}/{e['config_hash']}] missing '{field}'")
                row[field] = value

            write_row(row)
            n_done += 1
            print(f"  [{i + 1}/{len(entries)}] {e['config_hash'][:12]}...  "
                 f"budget={e['budget']}  MA={ma:.4f}  PR={pr:.4f}")
        except Exception as ex:
            n_failed += 1
            print(f"  [{i + 1}/{len(entries)}] {e['config_hash'][:12]}...  ERROR: {ex}")
        finally:
            if backbone is not None:
                del backbone
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return (n_done, n_skipped, n_failed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate privacy risk on independent singleHPO_results "
                    "checkpoints to check the hyperparameter->PR rules replicate.")
    p.add_argument("--output", default="validation_singlehpo.csv")
    p.add_argument("--adv-epochs", type=int, default=30,
                   help="Matches the convention used everywhere else in this project.")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Optional subset of singleHPO experiment names to scan "
                        "(default: all 6 registered).")
    p.add_argument("--max-budget-only", action="store_true",
                   help="Only evaluate configs whose own highest checkpoint equals this "
                        "experiment's max observed budget (skip pruned/undertrained "
                        "configs entirely, saving adversary-training GPU time). Default "
                        "keeps every config regardless of budget, per collect_hpo_dataset.py's convention.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_done_keys(output_path: str) -> list[dict]:
    """Existing rows from a prior (possibly interrupted) run of this same --output
    path, so they're neither lost nor recomputed. Returns the raw rows (to be
    rewritten verbatim) -- done_keys is derived from these by the caller."""
    if not os.path.exists(output_path):
        return []
    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Resuming: {len(rows)} rows already in {output_path}")
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    exp_names = args.experiments if args.experiments else list(SINGLE_EXPERIMENTS.keys())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    existing_rows = _load_done_keys(args.output)
    done_keys = {(r["source_dir"], r["config_hash"]) for r in existing_rows}

    # Rewrite the file starting with whatever already existed (verbatim), then
    # append new rows as they complete -- so a crash after this point never loses
    # more than the checkpoint currently in flight.
    f = open(args.output, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for r in existing_rows:
        writer.writerow(r)
    f.flush()

    def write_row(row: dict) -> None:
        writer.writerow(row)
        f.flush()

    total_done = total_skipped = total_failed = 0
    for exp_name in exp_names:
        print("=" * 70)
        print(f"[{exp_name}]")
        if exp_name not in SINGLE_EXPERIMENTS:
            print("  SKIPPED: not in registry")
            continue
        cfg = SINGLE_EXPERIMENTS[exp_name]
        print(f"  dataset={cfg['dataset']}  main_task={cfg['main_task']}  "
             f"sensitive_task={cfg['sensitive_task']}  backbone={cfg['backbone']}  "
             f"adv_config={_adv_config_path(cfg['counterpart'])}")

        n_done, n_skipped, n_failed = evaluate_experiment(
            exp_name, cfg, device, args.adv_epochs, write_row, done_keys,
            max_budget_only=args.max_budget_only)
        print(f"  {n_done} done, {n_skipped} already done (skipped), {n_failed} failed")
        total_done += n_done
        total_skipped += n_skipped
        total_failed += n_failed

    f.close()
    print("=" * 70)
    if total_failed:
        print(f"{total_failed} checkpoint(s) failed -- rerun the same command to retry "
             f"just those (everything else is skipped as already done).")
    print(f"Done. {len(existing_rows) + total_done} total rows in {args.output} "
         f"({total_done} new this run, {total_skipped} already done, {total_failed} failed)")


if __name__ == "__main__":
    main()
