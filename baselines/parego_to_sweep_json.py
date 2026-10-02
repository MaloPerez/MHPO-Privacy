"""
Convert a ParEGO run's checkpoints into the JSON format produced by pareto_sweep.py,
so that plot_pareto_curves.py can overlay ParEGO results on the baseline Pareto curves.

By default, reuses each checkpoint's stored "1 - main_task_accuracy" and the
privacy_risk computed during the original HPO search. Those aren't directly
comparable to pareto_sweep.py's baselines (different accuracy metric, different
adversary/epochs), so pass --reeval to recompute balanced accuracy from the
checkpoint's backbone weights, and --reeval-pr to retrain a fresh adversary with
pareto_sweep.py's own _eval_privacy for an apples-to-apples PR.

Usage
-----
python baselines/parego_to_sweep_json.py \\
  --model-path results/my_run/model_checkpoints \\
  --sensitive-task race --dataset FairFace --main-task gender \\
  --backbone IR_SE_50 --embedding-size 512 \\
  --reeval --reeval-pr --adv-epochs 30 \\
  --output results/parego_gender_race.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

NUM_CLASS_DICT = {'gender': 2, 'race': 7, 'age': 9}


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def _find_highest_budget_checkpoint(config_dir: str) -> tuple[str | None, int]:
    pattern = re.compile(r'budget_(\d+)_checkpoint\.pth')
    best_budget, best_path = -1, None
    for fname in os.listdir(config_dir):
        m = pattern.match(fname)
        if m:
            budget = int(m.group(1))
            if budget > best_budget:
                best_budget = budget
                best_path = os.path.join(config_dir, fname)
    return best_path, best_budget


def _load_one(config_dir: str, sensitive_task: str, dataset: str, defense_name: str) -> dict | None:
    path, budget = _find_highest_budget_checkpoint(config_dir)
    if path is None:
        return None

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    one_minus_ma = ckpt.get('1 - main_task_accuracy')
    if one_minus_ma is None:
        warnings.warn(f"No '1 - main_task_accuracy' in {path}, skipping.")
        return None
    ma_top1 = 1.0 - float(one_minus_ma)

    pr_key = f'privacy_risk_{sensitive_task.lower()}'
    raw_pr = ckpt.get(pr_key)
    if raw_pr is None:
        for k, v in ckpt.items():
            if 'privacy_risk' in k.lower():
                raw_pr = v
                warnings.warn(f"Used key '{k}' instead of '{pr_key}' in {path}.")
                break
    if raw_pr is None:
        warnings.warn(f"No privacy_risk key found in {path}, skipping.")
        return None

    num_class = 2 if dataset == 'CelebA' else NUM_CLASS_DICT.get(sensitive_task.lower(), 2)
    pr = max(0.0, float(raw_pr) - 1.0 / num_class)

    return {
        'defense': defense_name,
        'param_name': 'config_hash',
        'param_value': os.path.basename(config_dir),
        'budget': budget,
        'MA': ma_top1,
        'PR': pr,
    }


# ---------------------------------------------------------------------------
# Optional: recompute balanced MA and/or freshly re-measured PR using each
# checkpoint's own backbone (+ head, for MA) weights.
# ---------------------------------------------------------------------------

def _reeval(results: list[dict], model_path: str, cfg: dict, device,
           reeval_ma: bool, reeval_pr: bool, save_callback=None,
           existing_by_hash: dict | None = None) -> list[dict]:
    """
    reeval_ma: recompute MA as balanced accuracy of the stored head on the val set.
    reeval_pr: recompute PR by training a fresh adversary via pareto_sweep.py's
               _eval_privacy, so it's directly comparable across defenses. Configs
               already reevaluated in a prior run are skipped, and save_callback (if
               given) runs after every config so a kill mid-run loses no progress.
    """
    from sklearn.metrics import balanced_accuracy_score
    from model_resnet import ResNet_50, ResNet_101, ResNet_152
    from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
    from baselines import Encoder, Public_Classifier
    from metrics import ArcFace, CosFace, Softmax
    from pareto_sweep import _load_datasets, _resolve_tasks, _eval_privacy, _make_loader, _get_adv_config

    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 128), shuffle=False)
    adv_cfg = _get_adv_config(cfg) if reeval_pr else None

    backbone_cls = {
        'ResNet_50': ResNet_50, 'ResNet_101': ResNet_101, 'ResNet_152': ResNet_152,
        'IR_50': IR_50, 'IR_101': IR_101, 'IR_152': IR_152,
        'IR_SE_50': IR_SE_50, 'IR_SE_101': IR_SE_101, 'IR_SE_152': IR_SE_152,
        'Baseline': lambda s, e: Encoder(),
    }[cfg['backbone']]

    num_class_main = 2 if cfg['dataset'] == 'CelebA' else NUM_CLASS_DICT[cfg['main_task'].lower()]
    gpu_id = [int(str(device).split(':')[-1])] if 'cuda' in str(device) else [0]

    def _build_head(name):
        return {
            'ArcFace': ArcFace(in_features=cfg['embedding_size'], out_features=num_class_main, device_id=gpu_id),
            'CosFace': CosFace(in_features=cfg['embedding_size'], out_features=num_class_main, device_id=gpu_id),
            'Linear': Softmax(in_features=cfg['embedding_size'], out_features=num_class_main, device_id=gpu_id),
            'Baseline': Public_Classifier(input=cfg['embedding_size'], output=num_class_main),
        }[name]

    existing_by_hash = existing_by_hash or {}
    updated = []
    n_skipped = 0
    for i, r in enumerate(results):
        config_hash = r['param_value']
        prev = existing_by_hash.get(config_hash)
        if prev is not None and (not reeval_ma or prev.get('_ma_reevaluated')) \
                and (not reeval_pr or prev.get('_pr_reevaluated')):
            updated.append(prev)
            n_skipped += 1
            continue

        config_dir = os.path.join(model_path, config_hash)
        ckpt_path, _ = _find_highest_budget_checkpoint(config_dir)
        if ckpt_path is None:
            updated.append(r)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = backbone_cls(cfg['input_size'], cfg['embedding_size']).to(device)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        backbone.eval()

        r = dict(r)
        log = []

        if reeval_ma:
            head_name = ckpt.get('head', cfg['head_name'])
            head = _build_head(head_name).to(device)
            head.load_state_dict(ckpt['head_state_dict'])
            head.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if cfg['dataset'] == 'CelebA':
                        imgs = batch[0].to(device)
                        labels = batch[1][:, main_key].reshape(-1)
                    else:
                        imgs = batch['img'].to(device)
                        labels = batch[main_key].reshape(-1)
                    feats = backbone(imgs)
                    if isinstance(head, (Softmax, Public_Classifier)):
                        logits = head(feats)
                    else:
                        logits = head(feats, labels.to(device))
                    all_preds += logits.topk(1)[1].cpu().tolist()
                    all_targets += labels.tolist()
            r['MA'] = balanced_accuracy_score(all_targets, all_preds)
            r['_ma_reevaluated'] = True
            log.append(f"MA={r['MA']:.4f}")

        if reeval_pr:
            r['PR'] = _eval_privacy(backbone, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
            r['_pr_reevaluated'] = True
            log.append(f"PR={r['PR']:.4f}")

        updated.append(r)
        print(f"  [{i + 1}/{len(results)}] {r['param_value'][:12]}...  " + "  ".join(log))
        if save_callback is not None:
            save_callback(updated)

    if n_skipped:
        print(f"  ({n_skipped} configs already reevaluated in a previous run, skipped)")

    return updated


def _write_output(results: list[dict], experiment: dict, path: str) -> None:
    output = {'experiment': experiment, 'results': results}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Convert ParEGO checkpoints to pareto_sweep JSON format')
    p.add_argument('--model-path', required=True,
                   help='Path to the model_checkpoints directory of the ParEGO run')
    p.add_argument('--sensitive-task', required=True,
                   help='Sensitive task name (e.g. race, gender, age)')
    p.add_argument('--dataset', default='FairFace', choices=['FairFace', 'CelebA'])
    p.add_argument('--main-task', default='gender')
    p.add_argument('--backbone', default='IR_SE_50')
    p.add_argument('--head-name', default='Linear',
                   help='Fallback head name if not stored in checkpoint')
    p.add_argument('--embedding-size', type=int, default=512)
    p.add_argument('--input-size', nargs=2, type=int, default=[112, 112])
    p.add_argument('--defense-name', default='m_hpo',
                   help='Label used as the "defense" field in the JSON '
                        '(matches plot_pareto_curves.py\'s style dict, e.g. "m_hpo" or "m_hpo_pca")')
    p.add_argument('--reeval', action='store_true',
                   help='Reload each checkpoint\'s backbone weights and recompute balanced MA on the val set')
    p.add_argument('--reeval-pr', action='store_true',
                   help='Train a fresh adversary on each checkpoint\'s embeddings and recompute PR, '
                        'matching pareto_sweep.py\'s _eval_privacy (instead of using the privacy_risk '
                        'stored in the checkpoint from the original HPO run)')
    p.add_argument('--adv-epochs', type=int, default=30,
                   help='Adversary training epochs when --reeval-pr is set')
    p.add_argument('--adv-config-path', default=None,
                   help='Path to pickled adversary config (see pareto_sweep.py); '
                        'only used when --reeval-pr is set')
    p.add_argument('--batch-size', type=int, default=128,
                   help='Batch size for the val-set forward pass (--reeval) and adversary '
                        'training/eval loaders (--reeval-pr)')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='parego_results.json')
    p.add_argument('--include-partial-budget', action='store_true',
                   help='Also reeval/report configs that were pruned early by successive halving '
                        '(did not reach the max budget found). Off by default: those checkpoints '
                        'never reached their final trained state, so reevaluating them just wastes '
                        'a full --adv-epochs adversary training per config for no useful data point.')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")

    config_dirs = sorted(
        os.path.join(args.model_path, d)
        for d in os.listdir(args.model_path)
        if os.path.isdir(os.path.join(args.model_path, d))
    )
    print(f"Found {len(config_dirs)} config directories in {args.model_path}")

    results = []
    for cdir in config_dirs:
        r = _load_one(cdir, args.sensitive_task, args.dataset, args.defense_name)
        if r is None:
            print(f"  Skipped: {os.path.basename(cdir)}")
            continue
        results.append(r)
        print(f"  {os.path.basename(cdir)[:12]}...  budget={r['budget']}  MA={r['MA']:.4f}  PR={r['PR']:.4f}")

    print(f"\nLoaded {len(results)} configs.")

    if results and not args.include_partial_budget:
        max_budget = max(r['budget'] for r in results)
        before = len(results)
        results = [r for r in results if r['budget'] == max_budget]
        if len(results) < before:
            print(f"Filtered to {len(results)}/{before} configs that reached the max budget "
                 f"found ({max_budget}); the rest were pruned early by successive halving and "
                 f"never reached their final trained state. Pass --include-partial-budget to "
                 f"keep them too.")

    experiment = {
        'main_task': args.main_task,
        'sensitive_task': args.sensitive_task,
        'dataset': args.dataset,
        'backbone': args.backbone,
        'source': 'parego',
    }

    if args.reeval or args.reeval_pr:
        print("Recomputing MA and/or PR using each checkpoint's backbone weights...")
        cfg = {
            'dataset': args.dataset,
            'backbone': args.backbone,
            'head_name': args.head_name,
            'embedding_size': args.embedding_size,
            'input_size': args.input_size,
            'main_task': args.main_task,
            'sensitive_task': args.sensitive_task,
            'adv_epochs': args.adv_epochs,
            'adv_config_path': args.adv_config_path,
            'batch_size': args.batch_size,
        }

        existing_by_hash = {}
        if os.path.exists(args.output):
            with open(args.output) as f:
                prev_output = json.load(f)
            for r in prev_output.get('results', []):
                existing_by_hash[r['param_value']] = r
            print(f"Resuming: found existing output at {args.output} with "
                 f"{len(existing_by_hash)} configs already recorded.")

        def _save_partial(partial_results):
            _write_output(partial_results, experiment, args.output)

        results = _reeval(results, args.model_path, cfg, torch.device(args.device),
                          reeval_ma=args.reeval, reeval_pr=args.reeval_pr,
                          save_callback=_save_partial, existing_by_hash=existing_by_hash)
        print("Done.")
    if not args.reeval:
        warnings.warn(
            "MA is derived from stored top-1 accuracy (not balanced accuracy). "
            "Pass --reeval for a fair comparison against pareto_sweep.py results."
        )
    if not args.reeval_pr:
        warnings.warn(
            "PR is the privacy_risk stored in the checkpoint from the original HPO run "
            "(different adversary/training procedure than pareto_sweep.py's baselines). "
            "Pass --reeval-pr to train a fresh adversary matching pareto_sweep.py's "
            "methodology for a fair comparison."
        )

    _write_output(results, experiment, args.output)

    ma_vals = [r['MA'] for r in results]
    pr_vals = [r['PR'] for r in results]
    print(f"\nSaved {len(results)} points to {args.output}")
    print(f"MA range: [{min(ma_vals):.4f}, {max(ma_vals):.4f}]")
    print(f"PR range: [{min(pr_vals):.4f}, {max(pr_vals):.4f}]")


if __name__ == '__main__':
    main()
