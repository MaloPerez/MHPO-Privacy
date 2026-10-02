"""
Pareto sweep: run each defense with multiple trade-off hyperparameter values and
record (MA, PR) pairs into a JSON file for Pareto curve plotting.

Primary privacy-utility knob swept per defense: noisy=var (Gaussian noise variance),
pca/s_pca=scale (Laplace noise scale, with/without Siamese fine-tuning),
cloak=cloak_coef (noise-increase regularizer strength), dl=epochs_auto_enc (PFRNet
disentangler training epochs), arl=lambda_val (adversarial loss weight).

MA = main-task balanced accuracy (higher better); PR = privacy risk =
adversary_balanced_accuracy - 1/num_classes (lower better).

Usage
-----
python baselines/pareto_sweep.py \\
  --embed-path checkpoints/backbone.pth --head-path checkpoints/head.pth \\
  --main-task GENDER --sensitive-task race --dataset FairFace \\
  --defenses noisy pca s_pca cloak dl \\
  --adv-config-path adv_configs/gender_race.pkl \\
  --output results/pareto_gender_race.json
"""

from __future__ import annotations

import argparse
import copy
import json
import wandb
import os
import pickle
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Dynamic path setup relative to this file
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

from data_loaders import FairFaceDataset, load_celebA_data
from baselines import Encoder, Public_Classifier, PCAReduction, NoiseInjection, PrivacyEmbedding, PCADecompression
from metrics import ArcFace, CosFace, Softmax
from focal import FocalLoss
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from utils import AverageMeter, accuracy
from MAC_model import BioTask_5Layers
from PFRNet import PFRNet

# Import core logic from the fixed baselines
sys.path.insert(0, os.path.dirname(__file__))
import SiameseFinetuning as _siamese
import CloakBaseline as _cloak
import DisantangledLearning as _dl

# ---------------------------------------------------------------------------
# Default sweep grids per defense
# ---------------------------------------------------------------------------
SWEEP_CONFIGS: dict[str, dict] = {
    'noisy': {
        'param': 'var',
        'values': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    },
    'pca': {
        'param': 'scale',
        'values': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        'finetuning': False,
    },
    's_pca': {
        'param': 'scale',
        'values': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        'finetuning': True,
    },
    'cloak': {
        'param': 'cloak_coef',
        'values': [0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
    },
    'dl': {
        'param': 'epochs_auto_enc',
        'values': [5, 10, 20, 40],
    },
    'arl': {
        'param': 'lambda_val',
        'values': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    },
    'hpo': {
        'param': 'checkpoint',
        'values': ['pretrained'],  # single point, no sweep
    },
}

NUM_CLASS_DICT = {'gender': 2, 'race': 7, 'age': 9}
CLASS_WEIGHT_IDX = {'gender': 0, 'race': 1, 'age': 2}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _transforms(input_size=(112, 112)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _load_datasets(cfg: dict):
    t = _transforms(cfg['input_size'])
    if cfg['dataset'] == 'CelebA':
        data_root = cfg.get('data_root', os.path.join(_ROOT, 'data'))
        train_ds, attr, weights_train = load_celebA_data(data_root, 'train', 'attr', t)
        val_ds, _, _ = load_celebA_data(data_root, 'valid', 'attr', t)
        index_label = dict(zip(attr, range(len(attr))))
        return train_ds, val_ds, index_label, weights_train
    else:
        data_root = cfg.get('data_root', os.path.join(_ROOT, 'data', 'FairFace',
                                                        'fairface-img-margin025-trainval'))
        csv_train = cfg.get('csv_train', os.path.join(_ROOT, 'data', 'FairFace',
                                                       'fairface_label_train.csv'))
        csv_val = cfg.get('csv_val', os.path.join(_ROOT, 'data', 'FairFace',
                                                   'fairface_label_val.csv'))
        train_ds = FairFaceDataset(data_root, csv_train, t)
        val_ds = FairFaceDataset(data_root, csv_val, t)
        return train_ds, val_ds, None, None


def _make_loader(dataset, batch_size=128, shuffle=True, num_workers=4):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, num_workers=num_workers, drop_last=False,
    )


def _build_backbone(name: str, input_size, embedding_size: int) -> nn.Module:
    d = {
        'ResNet_50': ResNet_50(input_size, embedding_size),
        'ResNet_101': ResNet_101(input_size, embedding_size),
        'ResNet_152': ResNet_152(input_size, embedding_size),
        'IR_50': IR_50(input_size, embedding_size),
        'IR_101': IR_101(input_size, embedding_size),
        'IR_152': IR_152(input_size, embedding_size),
        'IR_SE_50': IR_SE_50(input_size, embedding_size),
        'IR_SE_101': IR_SE_101(input_size, embedding_size),
        'IR_SE_152': IR_SE_152(input_size, embedding_size),
        'Baseline': Encoder(),
    }
    return d[name]


def _build_head(name: str, embedding_size: int, num_class: int, device) -> nn.Module:
    gpu_id = [int(str(device).split(':')[-1])] if 'cuda' in str(device) else [0]
    d = {
        'ArcFace': ArcFace(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
        'CosFace': CosFace(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
        'Linear': Softmax(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
        'Baseline': Public_Classifier(input=embedding_size, output=num_class).to(device),
    }
    return d[name]


def _load_pretrained(cfg: dict, num_class: int, device):
    backbone = _build_backbone(cfg['backbone'], cfg['input_size'], cfg['embedding_size']).to(device)
    head = _build_head(cfg['head_name'], cfg['embedding_size'], num_class, device).to(device)

    if cfg.get('checkpoint', False):
        ckpt = torch.load(cfg['embed_path'], map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
    else:
        backbone.load_state_dict(torch.load(cfg['embed_path'], map_location=device))
        head.load_state_dict(torch.load(cfg['head_path'], map_location=device))

    backbone.eval()
    head.eval()
    return backbone, head


def _resolve_tasks(cfg: dict, index_label):
    """Return (main_task_key, sensitive_task_key) in the format each dataset needs."""
    if cfg['dataset'] == 'CelebA':
        return index_label[cfg['main_task']], index_label[cfg['sensitive_task']]
    else:
        return cfg['main_task'].lower(), cfg['sensitive_task'].lower()


def _num_class(task_key, cfg: dict, index_label):
    if cfg['dataset'] == 'CelebA':
        return 2
    return NUM_CLASS_DICT[task_key]


def _celeba_class_weights(train_ds, task_key: int, device) -> torch.Tensor:
    """Balanced class weights for one CelebA attribute column (task_key is the
    numeric column index into train_ds.attr)."""
    y = train_ds.attr[:, task_key].numpy()
    w = compute_class_weight(class_weight='balanced', classes=np.asarray([0, 1]), y=y)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _get_adv_config(cfg: dict) -> dict:
    """Load a pre-computed adversary config or return a sensible default."""
    if cfg.get('adv_config_path'):
        with open(cfg['adv_config_path'], 'rb') as f:
            return pickle.load(f)
    # Default: lightweight SGD adversary
    return {
        'batch_size': 64,
        'head': 'Baseline',
        'loss': 'CrossEntropy',
        'optimizer': 'SGD',
        'lr_sgd': 0.05,
        'lr_adam': 1e-3,
    }


def _eval_utility(embed_net: nn.Module, train_ds, val_ds,
                  main_task_key, cfg: dict, device) -> float:
    """
    Train a fresh classifier on top of embed_net's features, then return
    balanced accuracy on the val set. Embedding size is inferred from a
    forward pass so this works for any defense including those that change
    the embedding dimension (e.g. DL / PFRNet).
    """
    bs = cfg.get('batch_size', 64)
    train_loader = _make_loader(train_ds, bs)
    val_loader = _make_loader(val_ds, bs, shuffle=False)
    num_class = 2 if cfg['dataset'] == 'CelebA' else _num_class(main_task_key, cfg, None)

    # Infer actual embedding size from one forward pass
    embed_net.eval()
    with torch.no_grad():
        sample = next(iter(train_loader))
        sample_imgs = (sample[0] if cfg['dataset'] == 'CelebA' else sample['img'])[:2].to(device)
        actual_emb_size = embed_net(sample_imgs).shape[1]

    head = Public_Classifier(input=actual_emb_size, output=num_class).to(device)
    opt = optim.Adam(head.parameters(), lr=1e-3)
    if cfg['dataset'] == 'CelebA':
        utility_weights = _celeba_class_weights(train_ds, main_task_key, device)
    else:
        utility_weights = train_ds.class_weights[CLASS_WEIGHT_IDX[main_task_key]].to(device)
    loss_fn = nn.CrossEntropyLoss(weight=utility_weights).to(device)

    head_epochs = cfg.get('utility_head_epochs', 10)
    for _ in range(head_epochs):
        head.train()
        embed_net.eval()
        for batch in train_loader:
            opt.zero_grad()
            if cfg['dataset'] == 'CelebA':
                imgs = batch[0].to(device)
                labels = batch[1][:, main_task_key].reshape(-1).to(device)
            else:
                imgs = batch['img'].to(device)
                labels = batch[main_task_key].reshape(-1).to(device)
            with torch.no_grad():
                feats = embed_net(imgs)
            loss_fn(head(feats), labels).backward()
            opt.step()

    all_preds, all_targets = [], []
    head.eval()
    with torch.no_grad():
        for batch in val_loader:
            if cfg['dataset'] == 'CelebA':
                imgs = batch[0].to(device)
                labels = batch[1][:, main_task_key].reshape(-1)
            else:
                imgs = batch['img'].to(device)
                labels = batch[main_task_key].reshape(-1)
            feats = embed_net(imgs)
            all_preds += head(feats).topk(1)[1].cpu().tolist()
            all_targets += labels.tolist()

    return balanced_accuracy_score(all_targets, all_preds)


def _eval_privacy(embed_net: nn.Module, train_ds, val_ds,
                  sensitive_task_key, cfg: dict, adv_cfg: dict, device) -> float:
    """
    Train an adversary classifier on embeddings and return PR.
    PR = adversary_balanced_accuracy - 1/num_classes  (paper eq. 2)
    """
    num_class = _num_class(sensitive_task_key, cfg, None)
    adv_epochs = cfg.get('adv_epochs', 30)
    bs = adv_cfg['batch_size']

    train_loader = _make_loader(train_ds, bs)
    val_loader = _make_loader(val_ds, bs, shuffle=False)

    # Build adversary
    adv_name = adv_cfg['head']
    gpu_id = [int(str(device).split(':')[-1])] if 'cuda' in str(device) else [0]
    adv_map = {
        '5Layer': BioTask_5Layers(in_features=cfg['embedding_size'], out_features=num_class).to(device),
        'Baseline': Public_Classifier(input=cfg['embedding_size'], output=num_class).to(device),
        'ArcFace': ArcFace(in_features=cfg['embedding_size'], out_features=num_class, device_id=gpu_id).to(device),
        'CosFace': CosFace(in_features=cfg['embedding_size'], out_features=num_class, device_id=gpu_id).to(device),
        'Linear': Softmax(in_features=cfg['embedding_size'], out_features=num_class, device_id=gpu_id).to(device),
    }
    adv_model = adv_map[adv_name]

    # Class-weighted CE: an unweighted adversary can collapse to the majority class on
    # imbalanced attributes (e.g. CelebA's Eyeglasses), masking real leakage as PR=0.
    if cfg['dataset'] == 'CelebA':
        weights = _celeba_class_weights(train_ds, sensitive_task_key, device)
    else:
        weights = train_ds.class_weights[CLASS_WEIGHT_IDX[sensitive_task_key]].to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)

    # Optimizer
    if adv_cfg['optimizer'] == 'SGD':
        opt = optim.SGD(adv_model.parameters(), lr=adv_cfg['lr_sgd'])
    else:
        opt = optim.Adam(adv_model.parameters(), lr=adv_cfg['lr_adam'])

    embed_net.eval()
    for _ in range(adv_epochs):
        adv_model.train()
        for batch in train_loader:
            opt.zero_grad()
            if cfg['dataset'] == 'CelebA':
                imgs = batch[0].to(device)
                target = batch[1][:, sensitive_task_key].reshape(-1).to(device)
            else:
                imgs = batch['img'].to(device)
                target = batch[sensitive_task_key].reshape(-1).to(device)

            with torch.no_grad():
                feats = embed_net(imgs)

            if isinstance(adv_model, (ArcFace, CosFace)):
                logits = adv_model(feats, target)
            else:
                logits = adv_model(feats)

            loss_fn(logits, target).backward()
            opt.step()

    # Evaluate
    all_preds, all_targets = [], []
    adv_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if cfg['dataset'] == 'CelebA':
                imgs = batch[0].to(device)
                target = batch[1][:, sensitive_task_key].reshape(-1)
            else:
                imgs = batch['img'].to(device)
                target = batch[sensitive_task_key].reshape(-1)

            feats = embed_net(imgs)
            if isinstance(adv_model, (ArcFace, CosFace)):
                logits = adv_model(feats, target.to(device))
            else:
                logits = adv_model(feats)

            all_preds += logits.topk(1)[1].cpu().tolist()
            all_targets += target.tolist()

    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    chance = 1.0 / num_class
    return max(0.0, bal_acc - chance)


# ---------------------------------------------------------------------------
# Per-defense run functions
# ---------------------------------------------------------------------------

def run_noisy(var: float, cfg: dict, device) -> tuple[float, float]:
    """Gaussian noise injected into embeddings; no retraining needed."""
    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)

    # Wrap backbone to inject Gaussian noise on embeddings
    class NoisyEmbedNet(nn.Module):
        def __init__(self, net, variance):
            super().__init__()
            self.net = net
            self.variance = variance

        def forward(self, x):
            feats = self.net(x)
            return feats + torch.randn_like(feats) * (self.variance ** 0.5)

    noisy_net = NoisyEmbedNet(backbone, var).to(device)
    noisy_net.eval()

    adv_cfg = _get_adv_config(cfg)

    ma = _eval_utility(noisy_net, train_ds, val_ds, main_key, cfg, device)
    pr = _eval_privacy(noisy_net, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
    return ma, pr


def run_pca(scale: float, finetuning: bool, cfg: dict, device) -> tuple[float, float]:
    """PCA dimensionality reduction + Laplace noise, with optional Siamese fine-tuning."""
    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)

    train_loader = _make_loader(train_ds, cfg.get('batch_size', 50))
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 50), shuffle=False)

    if finetuning:
        lr = cfg.get('lr', 1e-7)
        epochs_ft = cfg.get('epochs_fine_tuning', 3)
        # siamese_finetuning generates C(batch_size, 2) pairs per batch and stacks
        # them into a single forward pass — use a small batch size to avoid OOM.
        siamese_loader = _make_loader(train_ds, cfg.get('siamese_batch_size', 16))
        _siamese.siamese_finetuning(siamese_loader, backbone, head, cfg['dataset'],
                                    device, main_key, epochs_ft, lr)

    # Fit PCA on train features
    features = _siamese.extract_features(copy.deepcopy(backbone).to(device),
                                         train_loader, cfg['dataset'], device=str(device))
    pca_model = _siamese.fit_pca(features, output_dim=8)

    input_dim = pca_model.components_.shape[1]
    output_dim = pca_model.components_.shape[0]
    weight = torch.tensor(pca_model.components_.T, dtype=torch.float32).to(device)

    pca = PCAReduction(input_dim=input_dim, output_dim=output_dim, pretrained_matrix=weight).to(device)
    pca_decom = PCADecompression(weight).to(device)
    noise = NoiseInjection(scale=scale, device=device).to(device)

    pca_noise = PrivacyEmbedding(pca, noise)
    embed_net = nn.Sequential(backbone, pca_noise, pca_decom).to(device)
    embed_net.eval()

    adv_cfg = _get_adv_config(cfg)
    ma = _eval_utility(embed_net, train_ds, val_ds, main_key, cfg, device)
    pr = _eval_privacy(embed_net, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
    return ma, pr


def run_cloak(cloak_coef: float, cfg: dict, device) -> tuple[float, float]:
    """Cloak: learnable input-noise layer trained to preserve utility while increasing noise."""
    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)

    input_size = cfg['input_size']
    noise_layer = _cloak.NoisyInput(
        channels=3, height=input_size[0], width=input_size[1],
        min_scale=1e-4, max_scale=1.5, init_scale=1e-2, device=device,
    ).to(device)

    train_loader = _make_loader(train_ds, cfg.get('batch_size', 50))
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 50), shuffle=False)

    _cloak.train_cloak(
        noise_layer=noise_layer,
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        device=device,
        main_task_key=main_key,
        num_classes=num_class_main,
        coef=cloak_coef,
        lr=cfg.get('cloak_lr', 1e-2),
        epochs=cfg.get('cloak_epochs', 10),
        dataset=cfg['dataset'],
    )

    cloak_embed = _cloak.CloakEmbeddingNet(backbone=backbone, noise=noise_layer).to(device)
    cloak_embed.eval()

    adv_cfg = _get_adv_config(cfg)
    ma = _eval_utility(cloak_embed, train_ds, val_ds, main_key, cfg, device)
    pr = _eval_privacy(cloak_embed, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
    return ma, pr


def run_dl(epochs_auto_enc: int, cfg: dict, device) -> tuple[float, float]:
    """Disentangled Learning: PFRNet trained for a given number of epochs."""
    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)

    train_loader = _make_loader(train_ds, cfg.get('batch_size', 50))
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 50), shuffle=False)

    embedding_size = cfg['embedding_size']
    disantangler = PFRNet(
        input_dim=embedding_size,
        id_latent_dim=int(0.95 * embedding_size),
        attr_latent_dim=embedding_size - int(0.95 * embedding_size),
    ).to(device)

    _dl.train_PFRNET(disantangler, train_loader, cfg['dataset'], backbone, device,
                     sens_key, epochs_auto_enc)

    embed_dis = nn.Sequential(backbone, disantangler).to(device)
    embed_dis.eval()

    # For DL, the embedding passed to the adversary is the first output of PFRNet
    class DisentangledWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            out = self.net(x)
            # PFRNet returns (z_main, z_priv, x_recon); use z_main for utility
            return out[0] if isinstance(out, tuple) else out

    dl_embed = DisentangledWrapper(embed_dis).to(device)
    dl_embed.eval()

    # Override embedding_size for adversary to match z_main dimension
    adv_cfg_dl = _get_adv_config(cfg)
    cfg_dl = dict(cfg)
    cfg_dl['embedding_size'] = int(0.95 * embedding_size)

    ma = _eval_utility(dl_embed, train_ds, val_ds, main_key, cfg, device)
    pr = _eval_privacy(dl_embed, train_ds, val_ds, sens_key, cfg_dl, adv_cfg_dl, device)
    return ma, pr


def run_arl(lambda_val: float, cfg: dict, device) -> tuple[float, float]:
    """
    Adversarial Representation Learning: retrain from a pretrained backbone
    using the ARL alternating loss with a given LAMBDA.
    Runs a shortened training (10 adversarial epochs) for sweep efficiency.
    """
    from focal import FocalLoss
    from utils import separate_irse_bn_paras, separate_resnet_bn_paras, schedule_lr, AverageMeter, accuracy

    train_ds, val_ds, index_label, weights_train = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)
    num_class_sens = _num_class(sens_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)
    backbone.train()
    head.train()

    adv_model = BioTask_5Layers(in_features=cfg['embedding_size'], out_features=num_class_sens).to(device)

    loss_fn = nn.CrossEntropyLoss()
    if cfg['dataset'] == 'CelebA':
        adv_loss_fn = nn.CrossEntropyLoss()
    else:
        adv_loss_fn = nn.CrossEntropyLoss(
            weight=train_ds.class_weights[CLASS_WEIGHT_IDX[sens_key]].to(device)
        )

    # Separate BN params for weight decay
    backbone_name = cfg['backbone']
    if 'IR' in backbone_name:
        bn_params, other_params = separate_irse_bn_paras(backbone)
        _, head_other = separate_irse_bn_paras(head)
    else:
        bn_params, other_params = separate_resnet_bn_paras(backbone)
        _, head_other = separate_resnet_bn_paras(head)

    lr = cfg.get('lr', 0.1)
    wd = cfg.get('weight_decay', 5e-4)
    opt_enc = optim.SGD([{'params': other_params + head_other, 'weight_decay': wd},
                          {'params': bn_params}], lr=lr, momentum=0.9)
    opt_adv = optim.SGD(adv_model.parameters(), lr=cfg.get('adv_lr', 0.1))

    train_loader = _make_loader(train_ds, cfg.get('batch_size', 64))
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 64), shuffle=False)

    arl_epochs = cfg.get('arl_epochs', 10)
    for epoch in range(arl_epochs):
        backbone.train()
        head.train()
        adv_model.train()
        for i, batch in enumerate(train_loader):
            if cfg['dataset'] == 'CelebA':
                imgs = batch[0].to(device)
                labels = batch[1][:, main_key].reshape(-1).to(device)
                sens_labels = batch[1][:, sens_key].reshape(-1).to(device)
            else:
                imgs = batch['img'].to(device)
                labels = batch[main_key].reshape(-1).to(device)
                sens_labels = batch[sens_key].reshape(-1).to(device)

            feats = backbone(imgs)
            logits = head(feats) if isinstance(head, (Softmax, Public_Classifier)) else head(feats, labels)
            adv_logits = adv_model(feats)

            loss_main = loss_fn(logits, labels)
            loss_adv = adv_loss_fn(adv_logits, sens_labels)

            if i % 3 == 0:
                opt_enc.zero_grad()
                (loss_main - lambda_val * loss_adv).backward()
                opt_enc.step()
            elif i % 3 == 1:
                opt_adv.zero_grad()
                loss_adv.backward()
                opt_adv.step()
            else:
                opt_enc.zero_grad()
                loss_main.backward()
                opt_enc.step()

    backbone.eval()
    adv_cfg = _get_adv_config(cfg)
    ma = _eval_utility(backbone, train_ds, val_ds, main_key, cfg, device)
    pr = _eval_privacy(backbone, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
    return ma, pr


def run_hpo(cfg: dict, device) -> tuple[float, float]:
    """Evaluate the HPO-trained model from --embed-path as a single point (no sweep).
    Utility is measured with the trained head directly — no defense is applied."""
    train_ds, val_ds, index_label, _ = _load_datasets(cfg)
    main_key, sens_key = _resolve_tasks(cfg, index_label)
    num_class_main = _num_class(main_key, cfg, index_label)

    backbone, head = _load_pretrained(cfg, num_class_main, device)
    val_loader = _make_loader(val_ds, cfg.get('batch_size', 64), shuffle=False)

    # Use the trained head directly — no fresh head needed since no defense distorts embeddings
    backbone.eval()
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
            logits = head(feats) if isinstance(head, (Softmax, Public_Classifier)) else head(feats, labels.to(device))
            all_preds += logits.topk(1)[1].cpu().tolist()
            all_targets += labels.tolist()
    ma = balanced_accuracy_score(all_targets, all_preds)

    adv_cfg = _get_adv_config(cfg)
    pr = _eval_privacy(backbone, train_ds, val_ds, sens_key, cfg, adv_cfg, device)
    return ma, pr


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DEFENSE_RUNNERS = {
    'noisy': lambda val, cfg, device: run_noisy(val, cfg, device),
    'pca': lambda val, cfg, device: run_pca(val, False, cfg, device),
    's_pca': lambda val, cfg, device: run_pca(val, True, cfg, device),
    'cloak': lambda val, cfg, device: run_cloak(val, cfg, device),
    'dl': lambda val, cfg, device: run_dl(int(val), cfg, device),
    'arl': lambda val, cfg, device: run_arl(val, cfg, device),
    'hpo': lambda val, cfg, device: run_hpo(cfg, device),
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _completed_runs_count(results: list[dict], defense: str, param_value) -> int:
    """Number of successful repeats already recorded for this (defense, param_value)."""
    return sum(
        1 for r in results
        if r['defense'] == defense and r['param_value'] == param_value and r.get('MA') is not None
    )


def _save(output: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def sweep(cfg: dict, defenses: list[str], device, output_path: str, experiment: dict,
          defense_label: str | None = None, num_runs: int = 1, seed: int = 42) -> list[dict]:
    # Load existing results so the sweep is resumable
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get('results', [])
        print(f"Resuming: {len(results)} results already in {output_path}")

    wandb.config.update({**experiment, 'adv_epochs': cfg['adv_epochs'],
                         'defenses': defenses, 'output': output_path, 'num_runs': num_runs})

    output = {'experiment': experiment, 'results': results}
    step = len([r for r in results if r.get('MA') is not None])

    for defense in defenses:
        sweep_def = SWEEP_CONFIGS[defense]
        param_name = sweep_def['param']
        values = sweep_def['values']
        runner = DEFENSE_RUNNERS[defense]
        label = defense_label if defense_label else defense
        print(f"\n{'='*60}")
        print(f"Sweeping defense: {defense}  param: {param_name}")
        print(f"Values: {values}")
        print('=' * 60)
        for val in values:
            done = _completed_runs_count(results, label, val)
            remaining = max(0, num_runs - done)
            if remaining == 0:
                print(f"  {defense} | {param_name}={val} — skipping ({done}/{num_runs} runs already done)")
                continue
            if done > 0:
                print(f"  {defense} | {param_name}={val} — {done}/{num_runs} runs already done, "
                      f"running {remaining} more")
            for run_idx in range(done, done + remaining):
                _set_seed(seed + run_idx)
                print(f"  {defense} | {param_name}={val} | run {run_idx + 1}/{num_runs} ...", flush=True)
                try:
                    ma, pr = runner(val, cfg, device)
                    result = {
                        'defense': label,
                        'param_name': param_name,
                        'param_value': val,
                        'run_idx': run_idx,
                        'MA': float(ma),
                        'PR': float(pr),
                    }
                    print(f"    MA={ma:.4f}  PR={pr:.4f}")
                    wandb.log({
                        f'{defense}/MA': ma,
                        f'{defense}/PR': pr,
                        f'{defense}/{param_name}': val,
                        'defense': defense,
                        'param_value': val,
                        'run_idx': run_idx,
                        'MA': ma,
                        'PR': pr,
                    }, step=step)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    result = {
                        'defense': label,
                        'param_name': param_name,
                        'param_value': val,
                        'run_idx': run_idx,
                        'MA': None,
                        'PR': None,
                        'error': str(e),
                    }
                    wandb.log({
                        f'{defense}/error': str(e),
                        'defense': defense,
                        'param_value': val,
                        'run_idx': run_idx,
                    }, step=step)
                results.append(result)
                step += 1
                # Write after every run so the sweep is resumable
                _save(output, output_path)

    return results


def parse_args():
    p = argparse.ArgumentParser(description='Pareto sweep: run each defense with multiple hyperparameter values')
    p.add_argument('--embed-path', required=True, help='Path to pretrained backbone checkpoint')
    p.add_argument('--head-path', default=None, help='Path to pretrained head checkpoint (if not in same file as embed)')
    p.add_argument('--checkpoint', action='store_true',
                   help='Checkpoint file contains both backbone_state_dict and head_state_dict')
    p.add_argument('--backbone', default='IR_SE_50')
    p.add_argument('--head-name', default='Linear')
    p.add_argument('--embedding-size', type=int, default=512)
    p.add_argument('--input-size', nargs=2, type=int, default=[112, 112])
    p.add_argument('--main-task', default='GENDER')
    p.add_argument('--sensitive-task', default='race')
    p.add_argument('--dataset', default='FairFace', choices=['FairFace', 'CelebA'])
    p.add_argument('--data-root', default=None, help='Override default data root path')
    p.add_argument('--adv-config-path', default=None, help='Path to pickled adversary config')
    p.add_argument('--adv-epochs', type=int, default=30, help='Adversary training epochs during eval')
    p.add_argument('--utility-head-epochs', type=int, default=10,
                   help='Epochs to train the fresh utility head in _eval_utility')
    p.add_argument('--arl-epochs', type=int, default=10, help='ARL training epochs per sweep point')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--defenses', nargs='+', default=list(DEFENSE_RUNNERS.keys()),
                   choices=list(DEFENSE_RUNNERS.keys()),
                   help='Defenses to sweep (default: all)')
    p.add_argument('--defense-label', default=None,
                   help='Override the defense name written to the JSON (e.g. "hpo_gender_race"). '
                        'Useful when running a single defense and wanting a custom label for plotting.')
    p.add_argument('--output', default='pareto_results.json', help='Output JSON file path')
    p.add_argument('--num-runs', type=int, default=3,
                   help='Target number of independent repeats per (defense, param_value) point, '
                        'for confidence intervals. If --output already contains some completed runs '
                        'for a point (e.g. resuming), only the remaining runs needed to reach '
                        '--num-runs are executed. Backbone/head are still loaded once from '
                        '--embed-path each run (never retrained); only defense/adversary training '
                        'is repeated.')
    p.add_argument('--seed', type=int, default=42,
                   help='Base random seed; repeat i of a point uses seed + i.')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    wandb.init(project="Pareto Sweep")

    cfg = {
        'embed_path': args.embed_path,
        'head_path': args.head_path,
        'checkpoint': args.checkpoint,
        'backbone': args.backbone,
        'head_name': args.head_name,
        'embedding_size': args.embedding_size,
        'input_size': args.input_size,
        'main_task': args.main_task,
        'sensitive_task': args.sensitive_task,
        'dataset': args.dataset,
        'adv_config_path': args.adv_config_path,
        'adv_epochs': args.adv_epochs,
        'utility_head_epochs': args.utility_head_epochs,
        'arl_epochs': args.arl_epochs,
        'batch_size': args.batch_size,
    }
    if args.data_root:
        cfg['data_root'] = args.data_root

    experiment = {
        'main_task': args.main_task,
        'sensitive_task': args.sensitive_task,
        'dataset': args.dataset,
        'backbone': args.backbone,
    }

    sweep(cfg, args.defenses, device, args.output, experiment,
          defense_label=args.defense_label, num_runs=args.num_runs, seed=args.seed)
    print(f"\nDone. Results saved to {args.output}")
