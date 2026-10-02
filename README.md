# On the Impact of Hyper-Parameters on the Inference Time Privacy of Deep Neural Networks

Code for the paper studying how training hyper-parameters affect the risk of
*unintended feature leakage* at inference time: given the embeddings produced by a
DNN trained for one biometric task (e.g. gender classification), how much can an
adversary infer about an unrelated private attribute (e.g. ethnicity, age)? We use a
multi-objective, multi-fidelity hyper-parameter optimization (M-HPO) approach to
jointly maximize main-task accuracy and minimize this privacy risk, and compare
against five privacy-preserving baselines.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. See `requirements.txt` for exact/minimum package versions; for
an exact reproduction, capture a `pip freeze` from your training environment and pin
`smac`, `ConfigSpace`, `scikit-learn`, `scikit-image`, and `opencv-python` to match.

### Data

All dataset/log/checkpoint paths resolve relative to the repo root by default, or to
the `FACEEVOLVE_DATA_ROOT` environment variable if set:

```bash
export FACEEVOLVE_DATA_ROOT=/path/to/your/data/root   # optional, defaults to the repo root
```

- **FairFace**: download from the [FairFace dataset page](https://github.com/joojs/fairface)
  and place it at `$FACEEVOLVE_DATA_ROOT/data/FairFace/fairface-img-margin025-trainval/`,
  with `fairface_label_train.csv` and `fairface_label_val.csv` alongside it under
  `$FACEEVOLVE_DATA_ROOT/data/FairFace/`.
- **CelebA**: downloads automatically (via `torchvision.datasets.CelebA`) into
  `$FACEEVOLVE_DATA_ROOT/data/` the first time it's needed.

## Repository layout

| Path | What it is |
|---|---|
| `backbone/`, `head/`, `loss/` | Embedding-network backbones (IR-SE/ResNet), classification heads (ArcFace/CosFace/Softmax), loss functions |
| `data_processing/` | FairFace/CelebA/triplet dataset loaders |
| `configs/` | Per-experiment hyper-parameter/dataset configs (one `configurations[i]` entry per main-task/private-task/architecture/dataset combination) |
| `MultiObj/` | The HPO search itself: `ParEGOTain.py` (multi-objective M-HPO, the paper's core method), `GeneralHPO.py` (single-objective baseline), `adv_searcher.py` (adversary hyper-parameter search), `hyperparam_sweep.py` (controlled single-hyperparameter validation sweep), `train_embedding_net.py` |
| `MAC/` | `BioTask_*` adversary/classifier heads used as the privacy adversary and main-task classifier across the pipeline |
| `baselines/` | The five privacy-defense baselines (ARL, Cloak, Disentangled Learning, PCA+Laplace/Siamese fine-tuning), the Pareto sweep harness, and the HPO-dataset collection/validation/plotting scripts |
| `pareto/` | Statistical analysis of the ~1500 trained models (`analyze_hpo.py`, DerSimonian-Laird meta-analysis) and the forest-plot scripts |
| `util/` | Shared training/evaluation utilities, LFW-style verification |
| `test_and_priv.py` | Evaluate a trained checkpoint's main-task accuracy and privacy risk against a chosen sensitive attribute |

## Reproducing the paper's results

### 1. Multi-objective HPO (M-HPO) and the single-objective baseline

```bash
# M-HPO (accuracy + privacy), one run per configs/config_multi.py entry:
python MultiObj/ParEGOTain.py --config-file 1 --exp-path my_mhpo_run --n-trials 80 \
  --min-budget 5 --max-budget 120

# Single-objective HPO baseline (accuracy only), one run per configs/config_single.py entry:
python MultiObj/GeneralHPO.py --config-file 1 --exp-path my_singlehpo_run --n-trials 50 \
  --min-budget 5 --max-budget 120
```

Each run trains and checkpoints many embedding networks under `<exp-path>/model_checkpoints/`.
The 16 M-HPO runs (6 FairFace+VGG16, 6 FairFace+Inception-ResNet, 4 CelebA+VGG16) and
6 single-objective baselines used in the paper correspond to the entries in
`configs/config_multi.py` and `configs/config_single.py` respectively.

### 2. Validate the privacy-measurement methodology (Fig. 2)

```bash
python baselines/ablation_study.py \
  --head-path checkpoints/head.pth \
  --embed-path-1 checkpoints/embed_epoch13.pth \
  --embed-path-2 checkpoints/embed_epoch40.pth \
  --main-task GENDER --sensitive-task race
# (ablation_study_full.py for the fuller sweep; see --help for all options)
```

### 3. Baseline comparison and Pareto fronts (Figs. 3, 4, 9, Table IV)

```bash
# Sweep each baseline defense's trade-off hyperparameter:
python baselines/pareto_sweep.py \
  --embed-path checkpoints/backbone.pth --head-path checkpoints/head.pth \
  --main-task GENDER --sensitive-task race --dataset FairFace \
  --defenses noisy pca s_pca cloak dl arl \
  --adv-config-path adv_configs/gender_race.pkl \
  --output results/pareto_gender_race.json

# Convert the M-HPO checkpoints to the same JSON schema and merge both into one plot:
python baselines/parego_to_sweep_json.py --model-path my_mhpo_run/model_checkpoints \
  --sensitive-task race --dataset FairFace --output results/parego_gender_race.json
python baselines/plot_pareto_curves.py \
  --results results/pareto_gender_race.json results/parego_gender_race.json \
  --output figures/pareto_gender_race.pdf
```

### 4. Hyper-parameter -> privacy-risk analysis (Figs. 5-8)

```bash
# Consolidate every M-HPO checkpoint into one dataset:
python baselines/collect_hpo_dataset.py --output pareto/hpo_dataset.csv

# Run the DerSimonian-Laird meta-analysis:
python3 pareto/analyze_hpo.py

# Render the figures:
python3 pareto/plot_hpo_analysis.py            # Figs. 5, 6, 8 (weight decay / lr / batch size / loss)
python3 pareto/plot_categorical_key_forest.py  # Fig. 7 (head=CosFace, optimizer=SGD)
```

### 5. Replication on independent models (Table III) and transferability (Table II)

```bash
# Table III: replicate the hyperparameter -> PR rules on independent single-objective
# HPO checkpoints that were never part of the M-HPO analysis:
python3 baselines/validate_singlehpo.py --output validation_singlehpo.csv

# Table II: measure an M-HPO-trained model's leakage of an unrelated private attribute
# (the one it was never tuned against) with test_and_priv.py's --sensitive-task flag.
```

Pre-computed outputs for steps 4-5 are already checked into `pareto/hpo_dataset.csv`,
`pareto/hpo_analysis/summary/`, and `pareto/hpo_analysis/validation_singlehpo.csv`, so
the analysis and plotting steps can be run directly without re-running the full HPO
search (which trains on the order of 1500 models).

### 6. Single-hyperparameter validation sweep

```bash
python MultiObj/hyperparam_sweep.py --config-file 21 --sweep-param weight_decay \
  --exp-path validation-celeba-male-wearinghat --n-values 6 --num-runs 1
```

## Acknowledgments

This repository builds on the [face.evoLVe](https://zhaoj9014.github.io) project by
Jian Zhao for the embedding-network backbones. See `NOTICE` and `LICENSE` for details.
