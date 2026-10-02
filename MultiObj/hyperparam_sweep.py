"""
Controlled single-hyperparameter validation sweep.

Bypasses the SMAC optimizer and calls MultiObj.ParEGOTain.train_bio_task directly
with manually-constructed ConfigSpace.Configuration objects: every hyperparameter is
fixed at a sensible default except the one being swept (weight_decay, lr, or
optimizer). Used to validate a multi-objective HPO analysis finding
(pareto/analyze_hpo.py) on a combo never trained elsewhere in this project, without
the cost of a full new 80-100 trial search.

Output uses the same checkpoint format/hash scheme as every other ParEGO run, so
baselines/collect_hpo_dataset.py picks it up with no changes.

Usage
-----
python MultiObj/hyperparam_sweep.py \\
  --config-file 21 --sweep-param weight_decay \\
  --exp-path validation-celeba-male-wearinghat --n-values 6 --num-runs 1

(--sweep-param also accepts lr or optimizer; --n-values is ignored for optimizer,
which is categorical rather than a logspace grid.)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, InCondition, Integer
from smac.utils.configspace import get_config_hash

# Same sys.path setup ParEGOTain.py itself uses, resolved relative to the repo root
# so this script works regardless of where the repo is cloned.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

from config_multi import configurations
from ParEGOTain import train_bio_task  # noqa: E402  (import needs the sys.path setup above)

# ConfigSpace bounds, copied from ParEGOTain.py's own __main__ construction (kept in
# sync with that file -- if its search space ever changes, update here too) so a
# manually-built Configuration validates/hashes exactly like one SMAC would produce.
WEIGHT_DECAY_BOUNDS = (1e-6, 0.1)
LR_ADAM_BOUNDS = (1e-5, 1e-2)
LR_SGD_BOUNDS = (1e-4, 1)

# Both sweeps use SGD -- the multi-objective HPO analysis (pareto/analyze_hpo.py)
# found lr's effect on PR is far weaker/less reliable under Adam than under SGD, while
# weight_decay's effect holds under either optimizer, so SGD lets one sweep test the
# strongest available evidence for both.
SWEEP_OPTIMIZER = {"weight_decay": "SGD", "lr": "SGD"}

# Value used for whichever of weight_decay/lr isn't being swept. weight_decay=2e-4 and
# lr_sgd=0.05 are near-median, low-failure-rate points from hpo_dataset.csv (not each
# hyperparameter's own ConfigSpace default, which sits in a higher-failure-rate
# region). lr_adam=1e-3 is only used by ensure_adv_config()'s one-time bootstrap.
DEFAULTS = dict(weight_decay=2e-4, lr_adam=1e-3, lr_sgd=0.05, loss="Focal", batch_size=80)


def build_configspace(config_file: int) -> ConfigurationSpace:
    """Exact copy of ParEGOTain.py's __main__ ConfigurationSpace construction."""
    cs = ConfigurationSpace()
    lr_adam = Float("lr_adam", LR_ADAM_BOUNDS, default=1e-3, log=True)
    lr_sgd = Float("lr_sgd", (0.0001, 1), log=True, default=0.1)
    if configurations[config_file]['HEAD_NAME'] == 'Baseline':
        head = Categorical("head", ["Baseline", "ArcFace", "CosFace"], default="Baseline")
    else:
        head = Categorical("head", ["CosFace", "ArcFace", "Linear"], default="CosFace")
    weight_decay = Float("weight_decay", WEIGHT_DECAY_BOUNDS, default=0.01, log=True)
    loss = Categorical("loss", ["Focal", "CrossEntropy"], default="Focal")
    optimizer = Categorical("optimizer", ["Adam", "SGD"], default="SGD")
    batch_size = Integer("batch_size", (30, 100), default=80)

    use_lr_adam = InCondition(lr_adam, parent=optimizer, values=['Adam'])
    use_lr_sgd = InCondition(lr_sgd, parent=optimizer, values=['SGD'])

    cs.add_hyperparameters([lr_adam, lr_sgd, head, loss, optimizer, batch_size, weight_decay])
    cs.add_conditions([use_lr_adam, use_lr_sgd])
    return cs


def default_head(config_file: int) -> str:
    return 'Baseline' if configurations[config_file]['HEAD_NAME'] == 'Baseline' else 'CosFace'


def fixed_config(cs: ConfigurationSpace, config_file: int, optimizer: str = "Adam",
                 **overrides) -> Configuration:
    """A Configuration with every hyperparameter at a fixed default for the given
    optimizer, plus any overrides."""
    values = dict(
        head=default_head(config_file),
        loss=DEFAULTS["loss"],
        optimizer=optimizer,
        batch_size=DEFAULTS["batch_size"],
        weight_decay=DEFAULTS["weight_decay"],
    )
    if optimizer == "Adam":
        values["lr_adam"] = DEFAULTS["lr_adam"]
    else:
        values["lr_sgd"] = DEFAULTS["lr_sgd"]
    values.update(overrides)
    return Configuration(cs, values=values)


# optimizer is itself a validated finding from the multi-objective HPO analysis worth a
# direct generalization check (pareto/analyze_hpo.py: optimizer[SGD] -> higher PR, i.e. worse privacy, both total
# +0.232/75% sign-consistency/p=0.019 and net-of-MA +0.280/81%/p=8.0e-4) -- not just
# the confound weight_decay/lr's own sweeps have to hold fixed. Unlike weight_decay/lr
# it's categorical with exactly 2 levels, so there's no logspace grid to build --
# --n-values is ignored for this sweep-param.
OPTIMIZER_VALUES = ["Adam", "SGD"]


def sweep_values(param: str, n: int) -> list:
    if param == "optimizer":
        return OPTIMIZER_VALUES
    bounds = {"weight_decay": WEIGHT_DECAY_BOUNDS, "lr": LR_SGD_BOUNDS}[param]
    lo, hi = bounds
    return [float(v) for v in np.logspace(np.log10(lo), np.log10(hi), n)]


def make_config(cs: ConfigurationSpace, config_file: int, sweep_param: str, v) -> Configuration:
    """Build the Configuration for one sweep point (weight_decay, lr, or optimizer)."""
    if sweep_param == "optimizer":
        return fixed_config(cs, config_file, optimizer=v)
    param_key = "lr_sgd" if sweep_param == "lr" else "weight_decay"
    return fixed_config(cs, config_file, optimizer=SWEEP_OPTIMIZER[sweep_param], **{param_key: v})


def ensure_adv_config(cs: ConfigurationSpace, config_file: int, model_path: str,
                      adv_config_path: str, max_budget: int) -> None:
    """One-time bootstrap: train a fixed-default config briefly to build the adversary
    config this sweep's every value will reuse. No-op if adv_config.pkl already exists."""
    pkl_path = os.path.join(adv_config_path, "adv_config.pkl")
    if os.path.exists(pkl_path):
        print(f"Reusing existing adv_config at {pkl_path}")
        return
    print(f"Bootstrapping adv_config at {pkl_path} ...")
    config = fixed_config(cs, config_file)
    train_bio_task(config, 0, int(max_budget / 3), model_path=model_path,
                   adv_config_path=adv_config_path, checkpointing=True,
                   config_file=config_file, adv_search_mode=True)


def parse_args():
    p = argparse.ArgumentParser(description="Controlled single-hyperparameter validation sweep")
    p.add_argument("--config-file", type=int, required=True,
                   help="Index into configs/config_multi.py's `configurations` dict.")
    p.add_argument("--sweep-param", choices=["weight_decay", "lr", "optimizer"], required=True)
    p.add_argument("--exp-path", type=str, required=True,
                   help="Subdirectory under this config's RESULTS_ROOT for checkpoints/adv_config.")
    p.add_argument("--n-values", type=int, default=6)
    p.add_argument("--num-runs", type=int, default=1,
                   help="Repeats per value. NOTE: get_config_hash is value-based, not "
                        "seed-based -- with >1, each repeat after the first uses a "
                        "'_runN' suffixed model_path so checkpoints don't collide/"
                        "overwrite (the default num_runs=1 keeps the standard, "
                        "un-suffixed layout collect_hpo_dataset.py expects).")
    p.add_argument("--max-budget", type=int, default=120)
    p.add_argument("--checkpointing", type=bool, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_file = configurations[args.config_file]
    results_root = cfg_file["RESULTS_ROOT"]
    base_model_path = os.path.join(results_root, args.exp_path, "model_checkpoints")
    adv_config_path = os.path.join(results_root, args.exp_path, "adv_config")

    cs = build_configspace(args.config_file)

    # The bootstrap isn't survivable to skip past -- every sweep value needs the
    # resulting adv_config.pkl (train_bio_task raises "Adversary not found yet"
    # without it), so a failure here aborts the whole sweep with one clear message
    # instead of the same per-value error repeating N times below.
    try:
        ensure_adv_config(cs, args.config_file, base_model_path, adv_config_path, args.max_budget)
    except Exception as ex:
        print(f"\nFATAL: adv_config bootstrap failed: {ex}")
        print("Nothing was swept. Rerun the same command -- ensure_adv_config re-checks "
             "for an existing adv_config.pkl before retrying the bootstrap.")
        return

    values = sweep_values(args.sweep_param, args.n_values)
    optimizer_note = "" if args.sweep_param == "optimizer" else f", optimizer={SWEEP_OPTIMIZER[args.sweep_param]}"
    print(f"Sweeping {args.sweep_param} (config-file={args.config_file}, "
         f"exp-path={args.exp_path}{optimizer_note}) over "
         f"{values} x {args.num_runs} repeat(s)")

    n_ok = n_failed = 0
    failed_values = []
    for v in values:
        for run in range(args.num_runs):
            config = make_config(cs, args.config_file, args.sweep_param, v)
            model_path = base_model_path if args.num_runs == 1 else f"{base_model_path}_run{run}"
            v_str = v if isinstance(v, str) else f"{v:.3g}"
            print(f"\n{'=' * 60}")
            print(f"{args.sweep_param}={v_str}  run={run}  hash={get_config_hash(config)}")
            print("=" * 60)
            # A failure here (e.g. CUDA OOM, the same failure mode seen in this
            # project's pareto_sweep.py defense sweeps) used to kill the whole
            # remaining sweep. Now it's caught and logged, and the loop moves on to
            # the next value -- train_bio_task's own on-disk checkpoint means a
            # later rerun of this same command will skip every value that already
            # completed and only retry what's in failed_values below.
            try:
                train_bio_task(config, run, args.max_budget, model_path=model_path,
                               adv_config_path=adv_config_path, checkpointing=args.checkpointing,
                               config_file=args.config_file)
                n_ok += 1
            except Exception as ex:
                n_failed += 1
                failed_values.append((v, run))
                print(f"  ERROR: {ex}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed. Checkpoints under {base_model_path}[_runN]")
    if failed_values:
        print(f"Failed (value, run): {failed_values}")
        print("Rerun the same command to retry just these -- completed values are "
             "skipped automatically (train_bio_task sees their checkpoint already exists).")


if __name__ == "__main__":
    main()
