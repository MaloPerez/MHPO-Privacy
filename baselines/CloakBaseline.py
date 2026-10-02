"""Cloak baseline (input-noise cloaking) analogous to DisantangledLearning.py.

Loads a pretrained embedding backbone (and optionally a pretrained main-task head),
freezes it, then trains a learnable stochastic input-noise layer ("cloak") so that
main-task performance is preserved while injected noise magnitude is encouraged to
increase via a scale-regularizer. Evaluates both main-task accuracy and sensitive-task
leakage via aux_adversary_privacy_risk.

Reuses the same data-loading and adversary-evaluation code patterns as
DisantangledLearning.py for consistency across baselines.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import wandb

# --- Project imports (same as DisantangledLearning.py) ---
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

from adv_searcher import get_best_adv_config
from data_loaders import FairFaceDataset, load_celebA_data
from baselines import Encoder, Public_Classifier
from metrics import ArcFace, CosFace, Softmax
from focal import FocalLoss
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from utils import AverageMeter, accuracy
from MAC_model import BioTask_5Layers


def parse_args():
    p = argparse.ArgumentParser(description='Cloak baseline (learned stochastic input noise)')
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--data-path', type=str, default=f'{_ROOT}/MAC/data')
    p.add_argument('--head-path', type=str, default=None,
                   help='Optional: checkpoint for pretrained main-task head')
    p.add_argument('--embed-path', type=str, required=True,
                   help='Checkpoint for pretrained backbone/embedding model')
    p.add_argument('--adv-config-path', type=str, default=None)
    p.add_argument('--adv-epochs', type=int, default=50)
    p.add_argument('--embedding-size', type=int, default=512)
    p.add_argument('--backbone', type=str, default='IR_SE_50')
    p.add_argument('--head-name', type=str, default='Linear',
                   help='Head architecture name for adversary search (same as other baselines)')
    p.add_argument('--input-size', nargs='+', type=int, default=[112, 112])
    p.add_argument('--rgb-mean', nargs='+', type=float, default=[0.5, 0.5, 0.5])
    p.add_argument('--rgb-std', nargs='+', type=float, default=[0.5, 0.5, 0.5])
    p.add_argument('--main-task', type=str, default='GENDER')
    p.add_argument('--sensitive-task', type=str, default='race')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    p.set_defaults(checkpoint=False)
    p.add_argument('--dataset', type=str, default='FairFace', choices=['FairFace', 'CelebA'])
    p.add_argument('--n-trials', type=int, default=20)

    # Cloak training knobs
    p.add_argument('--cloak-epochs', type=int, default=2,
                   help='Number of epochs to train the cloak noise parameters')
    p.add_argument('--cloak-lr', type=float, default=1e-2)
    p.add_argument('--cloak-coef', type=float, default=10.0,
                   help='Strength of the noise-increase regularizer')
    p.add_argument('--min-scale', type=float, default=1e-4)
    p.add_argument('--max-scale', type=float, default=1.5)
    p.add_argument('--init-scale', type=float, default=1e-2)
    p.add_argument('--init-loc', type=float, default=0.0)

    return p.parse_args()


def load_model(model: nn.Module, state_dict):
    model.load_state_dict(state_dict)
    return model


def validation_bio(val_loader, device, backbone, main_task_key, head: nn.Module, dataset: str = 'FairFace'):
    top1 = AverageMeter()
    for batch_sample in tqdm(iter(val_loader)):
        if dataset == 'CelebA':
            inputs = batch_sample[0].to(device)
            labels = batch_sample[1][:, main_task_key].reshape((-1)).to(device)
        else:
            inputs = batch_sample['img'].to(device)
            labels = batch_sample[main_task_key].reshape((-1)).to(device)
        bs = inputs.size(0)

        features = backbone(inputs)
        if isinstance(head, (Softmax, Public_Classifier)):
            outputs = head(features)
        else:
            outputs = head(features, labels)

        prec1 = accuracy(outputs.data, labels)
        top1.update(prec1[0].item(), bs)
    return top1


def aux_adversary_privacy_risk(
    embedding_net,
    embedding_size,
    device,
    sensitive_task,
    adv_config,
    adv_epochs,
    DATASET,
    disantangled: bool = False,
):
    """Copied (verbatim) logic from DisantangledLearning.py for consistency."""

    adv_batch_size = adv_config["batch_size"]
    adv_name = adv_config['head']
    adv_loss_name = adv_config['loss']
    adv_optim_name = adv_config['optimizer']

    transforms_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if DATASET == 'CelebA':
        DATA_ROOT = f'{_ROOT}/data'
    else:
        DATA_ROOT = f'{_ROOT}/data/FairFace/fairface-img-margin025-trainval'
        CSV_TRAIN_PATH = f'{_ROOT}/data/FairFace/fairface_label_train.csv'
        CSV_VAL_PATH = f'{_ROOT}/data/FairFace/fairface_label_val.csv'

    if DATASET == 'CelebA':
        train_dataset, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", transforms_aux)
        val_dataset, _, weights_val = load_celebA_data(DATA_ROOT, "valid", "attr", transforms_aux)
        WEIGHTS_SENSITIVE = weights_train[sensitive_task].to(device)
    else:
        train_dataset = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, transforms_aux)
        val_dataset = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, transforms_aux)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=adv_batch_size, sampler=None, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=adv_batch_size, sampler=None, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    NUM_CLASS_DICT = {'gender': 2, 'race': 7, 'age': 9}
    CLASS_WEIGHT_IDX = {'gender': 0, 'race': 1, 'age': 2}

    if DATASET == 'CelebA':
        NUM_CLASS = 2
    else:
        NUM_CLASS = NUM_CLASS_DICT[sensitive_task]

    ADV_DICT = {
        '5Layer': BioTask_5Layers(in_features=embedding_size, out_features=NUM_CLASS).to(device),
        'Baseline': Public_Classifier(input=embedding_size, output=NUM_CLASS).to(device),
        'ArcFace': ArcFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
        'CosFace': CosFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
        'Linear': Softmax(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
    }
    adv_model = ADV_DICT[adv_name]

    if DATASET == 'CelebA':
        LOSS_DICT = {
            'Focal': FocalLoss(),
            'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS_SENSITIVE).to(device)
        }
    else:
        LOSS_DICT = {
            'Focal': FocalLoss(),
            'CrossEntropy': nn.CrossEntropyLoss(
                weight=train_dataset.class_weights[CLASS_WEIGHT_IDX[sensitive_task]].to(device)
            )
        }
    loss_crit_adv = LOSS_DICT[adv_loss_name]

    if adv_optim_name == 'SGD':
        LR = adv_config["lr_sgd"]
        optim_adv = optim.SGD(adv_model.parameters(), lr=LR)
    else:
        LR = adv_config["lr_adam"]
        optim_adv = optim.Adam(adv_model.parameters(), lr=LR)

    for _e in range(adv_epochs):
        for batch in train_loader:
            adv_model.train()
            optim_adv.zero_grad()

            if DATASET == 'CelebA':
                imgs = batch[0].to(device)
                target = batch[1][:, sensitive_task].reshape((-1)).to(device)
            else:
                imgs = batch['img'].to(device)
                target = batch[sensitive_task].reshape((-1)).to(device)

            with torch.no_grad():
                if disantangled:
                    input_adv = embedding_net(imgs)[0].to(device)
                else:
                    input_adv = embedding_net(imgs).to(device)

            if isinstance(adv_model, (ArcFace, CosFace)):
                forward_adv = adv_model.forward(input_adv, target).to(device)
            else:
                forward_adv = adv_model.forward(input_adv).to(device)

            loss_adv = loss_crit_adv.forward(forward_adv, target)
            loss_adv.backward()
            optim_adv.step()

    all_preds, all_targets = [], []
    for batch in val_loader:
        adv_model.eval()

        if DATASET == 'CelebA':
            imgs = batch[0].to(device)
            target = batch[1][:, sensitive_task].reshape((-1)).to(device)
        else:
            imgs = batch['img'].to(device)
            target = batch[sensitive_task].reshape((-1)).to(device)

        with torch.no_grad():
            if disantangled:
                input_adv = embedding_net(imgs)[0].to(device)
            else:
                input_adv = embedding_net(imgs).to(device)

        if isinstance(adv_model, (CosFace, ArcFace)):
            forward_adv = adv_model(input_adv, target).to(device)
        else:
            forward_adv = adv_model(input_adv).to(device)

        all_preds += forward_adv.topk(1)[1].cpu().tolist()
        all_targets += target.tolist()

    bal_acc_adv = balanced_accuracy_score(all_targets, all_preds)
    # PR = adversary_balanced_accuracy - chance_level (eq. 2 in paper)
    chance_level = 1.0 / NUM_CLASS
    return bal_acc_adv - chance_level


class NoisyInput(nn.Module):
    """Learnable per-pixel Gaussian noise: x -> x + (loc + scale * eps).

    scale is constrained to [min_scale, max_scale] via a tanh reparameterization.
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        min_scale: float = 1e-4,
        max_scale: float = 1.5,
        init_loc: float = 0.0,
        init_scale: float = 1e-2,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        shape = (1, channels, height, width)

        locs = torch.full(shape, float(init_loc))
        # inverse of scale() init is messy; approximate by mapping init_scale into rhos.
        # We'll initialize rhos so tanh(rhos) roughly matches desired scale fraction.
        frac = (float(init_scale) - self.min_scale) / max(1e-12, (self.max_scale - self.min_scale))
        frac = float(np.clip(frac, 1e-6, 1 - 1e-6))
        # frac = (1+tanh(rho))/2 => tanh(rho) = 2*frac - 1
        rho0 = float(np.arctanh(2 * frac - 1))
        rhos = torch.full(shape, rho0)

        if device is not None:
            locs = locs.to(device)
            rhos = rhos.to(device)

        self.locs = nn.Parameter(locs)
        self.rhos = nn.Parameter(rhos)

    def scales(self) -> torch.Tensor:
        # (1+tanh)/2 maps to [0,1]
        unit = (1.0 + torch.tanh(self.rhos)) / 2.0
        return unit * (self.max_scale - self.min_scale) + self.min_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(x)
        return x + (self.locs + self.scales() * eps)


class CloakEmbeddingNet(nn.Module):
    """Applies learned stochastic noise to input, then runs frozen backbone."""

    def __init__(self, backbone: nn.Module, noise: NoisyInput):
        super().__init__()
        self.backbone = backbone
        self.noise = noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_noisy = self.noise(x)
        return self.backbone(x_noisy)


def _freeze(module: nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def train_cloak(
    noise_layer: NoisyInput,
    backbone: nn.Module,
    head: nn.Module,
    train_loader,
    device,
    main_task_key: str,
    num_classes: int,
    coef: float,
    lr: float,
    epochs: int,
    dataset: str,
):
    """Train only the noise parameters to preserve main-task performance while increasing noise scale."""

    _freeze(backbone)
    _freeze(head)
    noise_layer.train()

    # Main-task criterion
    if dataset == 'CelebA':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        # For FairFace, we can use unweighted CE by default (weights handled elsewhere if needed).
        criterion = nn.CrossEntropyLoss().to(device)

    opt = optim.Adam(noise_layer.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        n = 0
        for batch in tqdm(iter(train_loader), desc=f"Cloak train {epoch+1}/{epochs}"):
            opt.zero_grad()

            if dataset == 'CelebA':
                imgs = batch[0].to(device)
                labels = batch[1][:, main_task_key].reshape((-1)).to(device)
            else:
                imgs = batch['img'].to(device)
                labels = batch[main_task_key].reshape((-1)).to(device)

            imgs_noisy = noise_layer(imgs)
            feats = backbone(imgs_noisy)

            if isinstance(head, (Softmax, Public_Classifier)):
                logits = head(feats)
            else:
                logits = head(feats, labels)

            # If head is ArcFace/CosFace, logits are still class logits; CE works.
            loss_main = criterion(logits, labels)
            # Encourage higher noise magnitude (same spirit as UTKFace notebook):
            # minimize loss_main - coef * log(mean(scale))
            mean_scale = torch.mean(noise_layer.scales())
            loss = loss_main - float(coef) * torch.log(mean_scale + 1e-12)

            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu())
            n += 1

        if n:
            print(f"Epoch {epoch+1}: loss={running_loss/n:.4f}  mean_scale={float(mean_scale.detach().cpu()):.6f}")


def main():
    args = parse_args()
    wandb.init(project="Cloak Learning")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    embed_path = args.embed_path
    head_path = args.head_path
    backbone_name = args.backbone
    head_name = args.head_name
    input_size = args.input_size
    rgb_mean = args.rgb_mean
    rgb_std = args.rgb_std
    embedding_size = args.embedding_size
    main_task = args.main_task
    sensitive_task = args.sensitive_task
    checkpoint = args.checkpoint
    adv_epochs = args.adv_epochs
    DATASET = args.dataset
    n_trials = args.n_trials

    transforms_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if DATASET == 'CelebA':
        DATA_ROOT = f'{_ROOT}/data'
        train_dataset, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", transforms_aux)
        val_dataset, _, _ = load_celebA_data(DATA_ROOT, "valid", "attr", transforms_aux)
        INDEX_LABEL = dict(zip(attr, range(len(attr))))
        main_task_idx = INDEX_LABEL[main_task]
        sensitive_task_idx = INDEX_LABEL[sensitive_task]
        main_task_key = main_task_idx
        sensitive_task_key = sensitive_task_idx
    else:
        DATA_ROOT = f'{_ROOT}/data/FairFace/fairface-img-margin025-trainval'
        CSV_TRAIN_PATH = f'{_ROOT}/data/FairFace/fairface_label_train.csv'
        CSV_VAL_PATH = f'{_ROOT}/data/FairFace/fairface_label_val.csv'
        train_dataset = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, transforms_aux)
        val_dataset = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, transforms_aux)
        main_task_key = main_task.lower()  # e.g., 'gender'
        sensitive_task_key = sensitive_task  # already lower (e.g., 'race')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=None, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=None, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    BACKBONE_DICT = {
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

    NUM_CLASS_DICT = {'AGE': 9, 'RACE': 7, 'GENDER': 2}
    if DATASET == 'CelebA':
        NUM_CLASS_MAIN = 2
    else:
        NUM_CLASS_MAIN = NUM_CLASS_DICT[main_task]

    backbone = BACKBONE_DICT[backbone_name].to(device)
    print("embed path:", embed_path)
    if checkpoint:
        ckpt = torch.load(embed_path, map_location=device)
        state_dict_embed = ckpt['backbone_state_dict']
    else:
        state_dict_embed = torch.load(embed_path, map_location=device)
    load_model(backbone, state_dict_embed)

    # Main-task head (optional but recommended): must map embedding -> main classes.
    # If not provided, we create a simple linear head and train it briefly on frozen backbone.
    if head_path is not None:
        # Heuristic: checkpoint could be raw state dict or dict with 'head_state_dict'
        raw = torch.load(head_path, map_location=device)
        if isinstance(raw, dict) and 'head_state_dict' in raw:
            head_state = raw['head_state_dict']
        else:
            head_state = raw
        head = Public_Classifier(input=embedding_size, output=NUM_CLASS_MAIN).to(device)
        load_model(head, head_state)
        print("Loaded pretrained head from:", head_path)
    else:
        head = Public_Classifier(input=embedding_size, output=NUM_CLASS_MAIN).to(device)
        _freeze(backbone)
        head.train()
        opt_head = optim.Adam(head.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss().to(device)
        # quick warmup training so cloak has a stable target
        for e in range(1):
            for batch in tqdm(iter(train_loader), desc="Warmup head"):
                opt_head.zero_grad()
                if DATASET == 'CelebA':
                    imgs = batch[0].to(device)
                    labels = batch[1][:, main_task_key].reshape((-1)).to(device)
                else:
                    imgs = batch['img'].to(device)
                    labels = batch[main_task_key].reshape((-1)).to(device)
                with torch.no_grad():
                    feats = backbone(imgs)
                logits = head(feats)
                loss = crit(logits, labels)
                loss.backward()
                opt_head.step()
        _freeze(head)

    # --- Cloak: learnable noise on the *input image* ---
    c, h, w = 3, input_size[0], input_size[1]
    noise_layer = NoisyInput(
        channels=c,
        height=h,
        width=w,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        init_loc=args.init_loc,
        init_scale=args.init_scale,
        device=device,
    ).to(device)

    train_cloak(
        noise_layer=noise_layer,
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        device=device,
        main_task_key=main_task_key,
        num_classes=NUM_CLASS_MAIN,
        coef=args.cloak_coef,
        lr=args.cloak_lr,
        epochs=args.cloak_epochs,
        dataset=DATASET,
    )

    cloak_embed = CloakEmbeddingNet(backbone=backbone, noise=noise_layer).to(device)
    cloak_embed.eval()

    # --- Adversary hyperparameter search ---
    n_trials_adv = n_trials
    min_budget_adv = 10
    max_budget_adv = adv_epochs
    eta_adv = 3

    if args.adv_config_path is not None:
        with open(args.adv_config_path, 'rb') as f:
            main_config = pickle.load(f)
        adv_config = main_config
    else:
        # main task config
        main_config = get_best_adv_config(
            cloak_embed, main_task_key, head_name,
            n_trials_adv, min_budget_adv, max_budget_adv, eta_adv,
            embedding_size, DATA_ROOT, DATASET, disantangled=False
        )
        # sensitive task config
        adv_config = get_best_adv_config(
            cloak_embed, sensitive_task_key, head_name,
            n_trials_adv, min_budget_adv, max_budget_adv, eta_adv,
            embedding_size, DATA_ROOT, DATASET, disantangled=False
        )

    print("Main-task adversary config:", main_config)
    print("Sensitive-task adversary config:", adv_config)

    val_acc = aux_adversary_privacy_risk(
        cloak_embed, embedding_size, device,
        main_task_key, main_config, adv_epochs, DATASET, disantangled=False
    )
    privacy_risk = aux_adversary_privacy_risk(
        cloak_embed, embedding_size, device,
        sensitive_task_key, adv_config, adv_epochs, DATASET, disantangled=False
    )

    print("Cloak baseline — Val Accuracy (balanced acc):", val_acc)
    print("Cloak baseline — Privacy Risk (balanced acc):", privacy_risk)

    wandb.log({"Val Accuracy": val_acc,
               "Privacy Risk": privacy_risk})


if __name__ == '__main__':
    main()
