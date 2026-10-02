"""
Scan singleHPO_results/ and multiHPO_results/ for a fixed set of experiments and report,
for each, the checkpoint to use for the Pareto-plot pipeline plus everything pareto_sweep.py
needs to load it correctly (backbone, embedding size, head architecture).

Ground truth is the checkpoint files themselves (model_checkpoints/<hash>/budget_*_checkpoint.pth),
not best_configs/Time_*.csv -- the CSV can be stale if an HPO run was resumed, whereas every
checkpoint stores its own '1 - main_task_accuracy', 'privacy_risk_<task>' (multi only) and 'head'.

singleHPO selection: among hashes trained to the max budget found, pick highest accuracy.
multiHPO selection: among hashes trained to the max budget found, compute the Pareto front
(MA vs PR per sensitive task) and report the whole front plus the max(MA - PR) "best balance"
point -- same heuristic plot_pareto_curves.py uses to annotate the recommended trade-off point.

Read-only, CPU-only (map_location='cpu'), no training.
"""

import glob
import json
import os
import re

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SINGLE_ROOT = os.path.join(_ROOT, "singleHPO_results")
MULTI_ROOT = os.path.join(_ROOT, "multiHPO_results")

NUM_CLASS_DICT = {"race": 7, "age": 9, "gender": 2}


def num_class_for(sensitive_task: str, dataset: str) -> int:
    if dataset == "CelebA":
        return 2
    return NUM_CLASS_DICT[sensitive_task.lower()]


# ---------------------------------------------------------------------------
# Experiment registry -- backbone/embedding_size/head_pool taken directly from
# configs/config_single.py and configs/config_multi.py, matched by MAIN_TASK/
# SENSITIVE_TASK, not guessed from folder names.
# ---------------------------------------------------------------------------

SINGLE_EXPERIMENTS = {
    "single-hpo-resnet-gender": dict(main_task="GENDER", dataset="FairFace",
                                     backbone="IR_SE_50", embedding_size=512,
                                     head_pool=["CosFace", "ArcFace", "Linear"]),
    "single-hpo-resnet-race": dict(main_task="RACE", dataset="FairFace",
                                   backbone="IR_SE_50", embedding_size=512,
                                   head_pool=["CosFace", "ArcFace", "Linear"]),
    "single-hpo-resnet-age": dict(main_task="AGE", dataset="FairFace",
                                  backbone="IR_SE_50", embedding_size=512,
                                  head_pool=["CosFace", "ArcFace", "Linear"]),
    "single-hpo-celeba-male": dict(main_task="Male", dataset="CelebA",
                                   backbone="Baseline", embedding_size=4608,
                                   head_pool=["Baseline", "ArcFace", "CosFace"]),
    "single-hpo-celeba-eyeglasses": dict(main_task="Eyeglasses", dataset="CelebA",
                                         backbone="Baseline", embedding_size=4608,
                                         head_pool=["Baseline", "ArcFace", "CosFace"]),
    "single-hpo-celeba-grayhair": dict(main_task="Gray_Hair", dataset="CelebA",
                                       backbone="Baseline", embedding_size=4608,
                                       head_pool=["Baseline", "ArcFace", "CosFace"]),
}

MULTI_EXPERIMENTS = {
    # ResNet / IR_SE_50, config_multi.py indices 13-18
    "hpo-resnet-gender-race-REBUTTAL": dict(main_task="GENDER", sensitive_task="race",
                                            dataset="FairFace", backbone="IR_SE_50",
                                            embedding_size=512,
                                            head_pool=["CosFace", "ArcFace", "Linear"]),
    "hpo-resnet-gender-age-REBUTTAL": dict(main_task="GENDER", sensitive_task="age",
                                           dataset="FairFace", backbone="IR_SE_50",
                                           embedding_size=512,
                                           head_pool=["CosFace", "ArcFace", "Linear"]),
    "hpo-resnet-race-gender-REBUTTAL": dict(main_task="RACE", sensitive_task="gender",
                                            dataset="FairFace", backbone="IR_SE_50",
                                            embedding_size=512,
                                            head_pool=["CosFace", "ArcFace", "Linear"]),
    "hpo-resnet-race-age-REBUTTAL": dict(main_task="RACE", sensitive_task="age",
                                         dataset="FairFace", backbone="IR_SE_50",
                                         embedding_size=512,
                                         head_pool=["CosFace", "ArcFace", "Linear"]),
    "hpo-resnet-age-gender-REBUTTAL": dict(main_task="AGE", sensitive_task="gender",
                                           dataset="FairFace", backbone="IR_SE_50",
                                           embedding_size=512,
                                           head_pool=["CosFace", "ArcFace", "Linear"]),
    "hpo-resnet-age-race-REBUTTAL": dict(main_task="AGE", sensitive_task="race",
                                         dataset="FairFace", backbone="IR_SE_50",
                                         embedding_size=512,
                                         head_pool=["CosFace", "ArcFace", "Linear"]),
    # CelebA baseline, config_multi.py indices 19, 20, 23, 24
    "hpo-baseline-celeba-eyeglasses-male": dict(main_task="Eyeglasses", sensitive_task="Male",
                                                dataset="CelebA", backbone="Baseline",
                                                embedding_size=4608,
                                                head_pool=["Baseline", "ArcFace", "CosFace"]),
    "hpo-baseline-celeba-grey-hair-male": dict(main_task="Gray_Hair", sensitive_task="Male",
                                               dataset="CelebA", backbone="Baseline",
                                               embedding_size=4608,
                                               head_pool=["Baseline", "ArcFace", "CosFace"]),
    "hpo-baseline-celeba-male-eyeglasses": dict(main_task="Male", sensitive_task="Eyeglasses",
                                                dataset="CelebA", backbone="Baseline",
                                                embedding_size=4608,
                                                head_pool=["Baseline", "ArcFace", "CosFace"]),
    "hpo-baseline-celeba-male-gray-hair": dict(main_task="Male", sensitive_task="Gray_Hair",
                                               dataset="CelebA", backbone="Baseline",
                                               embedding_size=4608,
                                               head_pool=["Baseline", "ArcFace", "CosFace"]),
}


def find_checkpoint_dirs(root: str, exp_name: str):
    """Return the model_checkpoints dir for an experiment, or None if missing."""
    candidates = [
        os.path.join(root, exp_name, "model_checkpoints"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def scan_hashes(model_checkpoints_dir: str):
    """
    For every config-hash subdirectory, load its own highest-budget checkpoint
    (map_location='cpu') and pull out the fields we need. Returns a list of dicts.
    """
    out = []
    for hash_dir in sorted(glob.glob(os.path.join(model_checkpoints_dir, "*"))):
        if not os.path.isdir(hash_dir):
            continue
        config_hash = os.path.basename(hash_dir)
        budgets = []
        for fname in os.listdir(hash_dir):
            m = re.match(r"budget_(\d+)_checkpoint\.pth", fname)
            if m:
                budgets.append((int(m.group(1)), fname))
        if not budgets:
            continue
        best_budget, best_fname = max(budgets, key=lambda t: t[0])
        ckpt_path = os.path.join(hash_dir, best_fname)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"    [skip] {ckpt_path}: {e}")
            continue
        out.append(dict(
            config_hash=config_hash,
            budget=best_budget,
            ckpt_path=ckpt_path,
            head=ckpt.get("head"),
            one_minus_ma=ckpt.get("1 - main_task_accuracy"),
            raw=ckpt,
        ))
    return out


def pareto_front(ma_vals, pr_vals):
    n = len(ma_vals)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ma_vals[j] >= ma_vals[i] and pr_vals[j] <= pr_vals[i]:
                if ma_vals[j] > ma_vals[i] or pr_vals[j] < pr_vals[i]:
                    dominated[i] = True
                    break
    return [i for i in range(n) if not dominated[i]]


def process_single(exp_name: str, cfg: dict):
    print("=" * 70)
    print(f"[single] {exp_name}")
    ckdir = find_checkpoint_dirs(SINGLE_ROOT, exp_name)
    if ckdir is None:
        print(f"  NOT FOUND: {os.path.join(SINGLE_ROOT, exp_name, 'model_checkpoints')}")
        return None
    entries = scan_hashes(ckdir)
    if not entries:
        print("  no checkpoints found")
        return None
    max_budget = max(e["budget"] for e in entries)
    finished = [e for e in entries if e["budget"] == max_budget]
    best = min(finished, key=lambda e: e["one_minus_ma"])
    ma = 1 - best["one_minus_ma"]
    print(f"  dataset={cfg['dataset']}  main_task={cfg['main_task']}  "
         f"backbone={cfg['backbone']}  embedding_size={cfg['embedding_size']}")
    print(f"  {len(entries)} configs scanned, {len(finished)} reached max budget ({max_budget})")
    print(f"  BEST: hash={best['config_hash']}  head={best['head']}  "
         f"top1_acc={ma:.4f}  path={best['ckpt_path']}")
    return dict(
        experiment=exp_name, role="single", dataset=cfg["dataset"],
        main_task=cfg["main_task"], backbone=cfg["backbone"],
        embedding_size=cfg["embedding_size"], head=best["head"],
        config_hash=best["config_hash"], budget=best["budget"],
        checkpoint_path=best["ckpt_path"], top1_acc=ma,
    )


def process_multi(exp_name: str, cfg: dict):
    print("=" * 70)
    print(f"[multi]  {exp_name}")
    ckdir = find_checkpoint_dirs(MULTI_ROOT, exp_name)
    if ckdir is None:
        print(f"  NOT FOUND: {os.path.join(MULTI_ROOT, exp_name, 'model_checkpoints')}")
        return None
    entries = scan_hashes(ckdir)
    if not entries:
        print("  no checkpoints found")
        return None
    max_budget = max(e["budget"] for e in entries)
    finished = [e for e in entries if e["budget"] == max_budget]

    sens_key = f"privacy_risk_{cfg['sensitive_task']}"
    num_class = num_class_for(cfg["sensitive_task"], cfg["dataset"])

    rows = []
    for e in finished:
        raw_pr = e["raw"].get(sens_key)
        if raw_pr is None:
            continue
        ma = 1 - e["one_minus_ma"]
        pr = max(0.0, float(raw_pr) - 1.0 / num_class)
        rows.append(dict(config_hash=e["config_hash"], head=e["head"], ma=ma, pr=pr,
                         ckpt_path=e["ckpt_path"]))

    if not rows:
        print(f"  no checkpoints at max budget ({max_budget}) have key '{sens_key}'")
        return None

    print(f"  dataset={cfg['dataset']}  main_task={cfg['main_task']}  "
         f"sensitive_task={cfg['sensitive_task']}  backbone={cfg['backbone']}  "
         f"embedding_size={cfg['embedding_size']}")
    print(f"  {len(entries)} configs scanned, {len(rows)} reached max budget ({max_budget}) "
         f"with a recorded {sens_key}")

    front_idx = pareto_front([r["ma"] for r in rows], [r["pr"] for r in rows])
    front = sorted((rows[i] for i in front_idx), key=lambda r: r["ma"])
    print(f"  Pareto front ({len(front)} points, sorted by MA):")
    for r in front:
        print(f"    hash={r['config_hash']}  head={r['head']:<10}  "
             f"MA={r['ma']:.4f}  PR={r['pr']:.4f}  MA-PR={r['ma']-r['pr']:.4f}")

    best = max(front, key=lambda r: r["ma"] - r["pr"])
    print(f"  BEST BALANCE (max MA-PR on front): hash={best['config_hash']}  "
         f"head={best['head']}  MA={best['ma']:.4f}  PR={best['pr']:.4f}  "
         f"path={best['ckpt_path']}")

    return dict(
        experiment=exp_name, role="multi", dataset=cfg["dataset"],
        main_task=cfg["main_task"], sensitive_task=cfg["sensitive_task"],
        backbone=cfg["backbone"], embedding_size=cfg["embedding_size"],
        head=best["head"], config_hash=best["config_hash"],
        checkpoint_path=best["ckpt_path"], MA=best["ma"], PR=best["pr"],
        pareto_front=[{"hash": r["config_hash"], "head": r["head"],
                      "MA": r["ma"], "PR": r["pr"]} for r in front],
    )


def main():
    summary = []
    for exp_name, cfg in SINGLE_EXPERIMENTS.items():
        r = process_single(exp_name, cfg)
        if r:
            summary.append(r)
    for exp_name, cfg in MULTI_EXPERIMENTS.items():
        r = process_multi(exp_name, cfg)
        if r:
            summary.append(r)

    print("=" * 70)
    print("JSON_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("JSON_SUMMARY_END")


if __name__ == "__main__":
    main()
