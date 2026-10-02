"""PCA + Laplace-noise privacy-defense baseline, optionally with Siamese
fine-tuning (S-PCA + Laplace)."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.decomposition import PCA
import math
import copy
import itertools
from sklearn.metrics import balanced_accuracy_score

import sys
import os

import functools
import pandas as pd
import csv
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from PFRNet import PFRNet
import pickle
import time
import numpy as np
import random

from ConfigSpace import (
    Configuration,
    Categorical,
    ConfigurationSpace,
    Float,
    InCondition,
    Integer
)

from smac import MultiFidelityFacade, Scenario
from smac.facade.abstract_facade import AbstractFacade
from sklearn.metrics import balanced_accuracy_score

from smac.multi_objective.parego import ParEGO
from smac.utils.configspace import get_config_hash

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

from adv_searcher import get_best_adv_config
from data_loaders import FairFaceDataset, load_celebA_data
from config_adv_training import configurations
import argparse
import copy
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from baselines import Encoder, Public_Classifier, PCAReduction, NoiseInjection, PrivacyEmbedding, PCADecompression
from metrics import ArcFace, CosFace, Softmax
from focal import FocalLoss
from utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from MAC_model import BioTask_2Layers, BioTask_1Layer, BioTask_5Layers

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Trains MAC model on selected public dataset')
    parser.add_argument('--batch-size', type=int, default='50',
                        help='balanced_ce or focal')
    parser.add_argument('--type-init', type=str, default="normal",
                        help='type of initialization for weights of mac model (default: normal)')
    parser.add_argument('--data-path', type=str, default=f'{_ROOT}/MAC/data',
                        help='path to the directory containing the data (default: deep-privacy/data)')
    parser.add_argument('--head-path', type=str)
    parser.add_argument('--embed-path', type=str)
    parser.add_argument('--adv-config-path', type=str)
    parser.add_argument('--adv-epochs', type=int, default='50')
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Size of embedding (default: 512)')
    parser.add_argument('--backbone', type=str, default='IR_SE_50',
                        help='name of the backbone to load')
    parser.add_argument('--head-name', type=str, default='Linear')
    parser.add_argument('--input-size', nargs='+', type=int, default=[112, 112])
    parser.add_argument('--rgb-mean', nargs='+', type=int, default=[0.5, 0.5, 0.5])
    parser.add_argument('--rgb-std', nargs='+', type=int, default=[0.5, 0.5, 0.5])
    parser.add_argument('--main-task', type=str, default='GENDER')
    parser.add_argument('--sensitive-task', type=str, default='race')
    # parser.add_argument('--checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=False)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--no-pca', dest='pca', action='store_false')
    parser.add_argument('--hpo', action='store_true')
    parser.add_argument('--no-hpo', dest='hpo', action='store_false')
    parser.add_argument('--finetuning', action='store_true')
    parser.add_argument('--no-finetuning', dest='finetuning', action='store_false')
    parser.add_argument('--dataset', type=str, default='FairFace')
    parser.add_argument('--epochs-fine-tuning', type=int, default='3')
    parser.add_argument('--n-trials', type=int, default='20')
    args = parser.parse_args()
    return args


def load_model(model, state_dict):
    model.load_state_dict(state_dict)
    return model


def extract_features(embed_net, dataloader, DATASET, device='cuda'):
    embed_net.eval()
    features = []

    i = 0

    with torch.no_grad():
        for batch_sample in dataloader:
            if DATASET == 'CelebA':
                inputs = batch_sample[0].to(device)
                if i < 3:
                    print(inputs)
            else:
                inputs = batch_sample['img'].to(device)

            f = embed_net(inputs)
            if i < 3:
                print(f)
            i+=1
            features.append(f.cpu())

    return torch.cat(features, dim=0)  # Shape: (N, feature_dim)


def fit_pca(features, output_dim):
    pca = PCA(n_components=output_dim)
    print('NaN ',np.any(np.isnan(features.numpy())))
    print('Inf ',np.any(np.isinf(features.numpy())))

    row_variances = features.var(dim=1)
    print("Min variance per row:", row_variances.min().item())
    print("Any zero-variance rows:", (row_variances == 0).any().item())
    col_variances = features.var(dim=0)
    print("Min variance per feature (column):", col_variances.min().item())
    try:
        pca.fit(features.numpy())
    except:
        pca.fit(features.numpy())
    return pca


def validation_bio(val_loader, DEVICE, BACKBONE, MAIN_TASK, HEAD, DATASET):
    batch = 0
    top1 = AverageMeter()

    for batch_sample in tqdm(iter(val_loader)):

        if DATASET == 'CelebA':
            inputs = batch_sample[0].to(DEVICE)
            labels = batch_sample[1][:, MAIN_TASK].reshape((-1)).to(DEVICE)
        else:
            inputs = batch_sample['img'].to(DEVICE)
            labels = batch_sample[MAIN_TASK.lower()].reshape((-1)).to(DEVICE)

        current_batch_size = inputs.size(0)

        features = BACKBONE(inputs)

        if isinstance(HEAD, Softmax) or isinstance(HEAD, Public_Classifier):
            outputs = HEAD(features)
        else:
            outputs = HEAD(features, labels)

        # measure val accuracy and val loss
        prec1 = accuracy(outputs.data, labels)
        top1.update(prec1[0].item(), current_batch_size)

        batch += 1  # batch index

    epoch_val_acc = top1

    return epoch_val_acc


def aux_adversary_privacy_risk(embedding_net, embedding_size, device, sensitive_task, adv_config, adv_epochs,
                               DATASET, disantangled=False):
    adv_batch_size = adv_config["batch_size"]
    adv_name = adv_config['head']
    adv_loss_name = adv_config['loss']
    adv_optim_name = adv_config['optimizer']

    transforms_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),  # smaller side resized
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

    sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=adv_batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=adv_batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    NUM_CLASS_DICT = {'gender': 2,
                      'race': 7,
                      'age': 9}
    CLASS_WEIGHT_IDX = {'gender': 0,
                        'race': 1,
                        'age': 2}

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
        LOSS_DICT = {'Focal': FocalLoss(weight=WEIGHTS_SENSITIVE).to(device),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS_SENSITIVE).to(device)}
    else:
        _sens_weight = train_dataset.class_weights[CLASS_WEIGHT_IDX[sensitive_task]].to(device)
        LOSS_DICT = {'Focal': FocalLoss(weight=_sens_weight),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=_sens_weight)}
    loss_crit_adv = LOSS_DICT[adv_loss_name]

    if adv_optim_name == 'SGD':
        LR = adv_config["lr_sgd"]
        optim_adv = optim.SGD(adv_model.parameters(), lr=LR)

    elif adv_optim_name == "Adam":
        LR = adv_config["lr_adam"]
        optim_adv = optim.Adam(adv_model.parameters(), lr=LR)

    for e in range(adv_epochs):

        for idx, batch in enumerate(train_loader):
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

            if idx == 0:
                print(input_adv)

            if isinstance(adv_model, ArcFace) or isinstance(adv_model, CosFace):
                forward_adv = adv_model.forward(input_adv, target).to(device)
            else:
                forward_adv = adv_model.forward(input_adv).to(device)

            loss_adv = loss_crit_adv.forward(forward_adv, target)

            loss_adv.backward()

            if idx == 0:
                for name, param in adv_model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad mean: {param.grad.abs().mean().item()}")
                    else:
                        print(f"{name} has no gradient!")

            optim_adv.step()

    all_preds = []
    all_targets = []

    for idx, batch in enumerate(val_loader):
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

        if isinstance(adv_model, CosFace) or isinstance(adv_model, ArcFace):
            forward_adv = adv_model(input_adv, target).to(device)
        else:
            forward_adv = adv_model(input_adv).to(device)

        all_preds = all_preds + forward_adv.topk(1)[1].cpu().tolist()
        all_targets = all_targets + target.tolist()

    bal_acc_adv = balanced_accuracy_score(all_targets, all_preds)
    # PR = adversary_balanced_accuracy - chance_level (eq. 2 in paper)
    chance_level = 1.0 / NUM_CLASS
    return bal_acc_adv - chance_level


def pair_inputs(batch_sample, dataset, device, main_task):
    if dataset == 'CelebA':
        inputs = batch_sample[0].to(device)
        labels = batch_sample[1][:, main_task].reshape((-1)).to(device)
    else:
        inputs = batch_sample['img'].to(device)
        labels = batch_sample[main_task.lower()].reshape((-1)).to(device)

    pairs_diff = []
    labels_diff = []
    pairs_same = []
    labels_same = []
    batch_size = inputs.size(0)

    for i, j in itertools.combinations(range(batch_size), 2):
        if labels[i] != labels[j]:
            pairs_diff.append((inputs[i], inputs[j]))
            labels_diff.append((labels[i], labels[j]))
        else:
            pairs_same.append((inputs[i], inputs[j]))
            labels_same.append((labels[i], labels[j]))

    return pairs_diff, pairs_same, labels_diff, labels_same


def contrastive_loss(f1, f2, label, margin=1.0):

    dist = F.pairwise_distance(f1, f2)
    loss = label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)
    return loss.mean()


def siamese_finetuning(train_loader, embed_net, head, dataset, device, main_class, epochs, lr):
    head_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(embed_net.parameters()) + list(head.parameters()), lr=lr)

    # Use train loader to get some data.
    total_epochs = epochs
    for epoch in range(0, total_epochs):
        print('Epoch:{}/{}', epoch + 1, total_epochs)
        for batch_sample in tqdm(iter(train_loader)):

            pairs_diff, pairs_same, labels_diff, labels_same = pair_inputs(batch_sample, dataset, device, main_class)

            all_pairs = pairs_same + pairs_diff
            labels_sim = torch.tensor([1] * len(pairs_same) + [0] * len(pairs_diff), dtype=torch.float32).to(device)
            all_labels = labels_diff + labels_same

            x1 = torch.stack([p[0] for p in all_pairs]).to(device)
            x2 = torch.stack([p[1] for p in all_pairs]).to(device)

            all_labels1 = torch.stack([p[0] for p in all_labels]).to(device)
            all_labels2 = torch.stack([p[1] for p in all_labels]).to(device)

            f1 = embed_net(x1)
            f2 = embed_net(x2)

            output1 = head(f1)
            output2 = head(f2)

            cont_loss = contrastive_loss(f1, f2, labels_sim)

            loss = cont_loss + head_loss(output1, all_labels1) + head_loss(output2, all_labels2)

            # .item() detaches from the autograd graph before logging, so wandb doesn't
            # pin these tensors' computation graphs in memory (backbone params are
            # trained here, so the graph is legitimately large).
            wandb.log({'Loss Contrastive': cont_loss.item(),
                       'Loss Total': loss.item()}
                      )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main(args):
    wandb.init(project="Siamese + PCA")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [0]

    # loading args
    batch_size = args.batch_size
    embed_path = args.embed_path
    head_path = args.head_path
    backbone = args.backbone
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
    finetuning = args.finetuning
    epochs_fine_tuning = args.epochs_fine_tuning
    n_trials = args.n_trials
    lr = args.lr

    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in
        transforms.ToPILImage(),
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean,
                             std=rgb_std),
    ])

    if DATASET == 'CelebA':
        DATA_ROOT = f'{_ROOT}/data'
    else:
        DATA_ROOT = f'{_ROOT}/data/FairFace/fairface-img-margin025-trainval'
        CSV_TRAIN_PATH = f'{_ROOT}/data/FairFace/fairface_label_train.csv'
        CSV_VAL_PATH = f'{_ROOT}/data/FairFace/fairface_label_val.csv'

    sampler = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    transforms_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),  # smaller side resized
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if DATASET == 'CelebA':
        train_dataset, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", transforms_aux)
        val_dataset, _, weights_val = load_celebA_data(DATA_ROOT, "valid", "attr", transforms_aux)

        INDEX_LABEL = dict(zip(attr, range(len(attr))))
        main_task = INDEX_LABEL[main_task]
        sensitive_task = INDEX_LABEL[sensitive_task]
    else:
        train_dataset = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, transforms_aux)
        val_dataset = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, transforms_aux)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    BACKBONE_DICT = {'ResNet_50': ResNet_50(input_size, embedding_size),
                     'ResNet_101': ResNet_101(input_size, embedding_size),
                     'ResNet_152': ResNet_152(input_size, embedding_size),
                     'IR_50': IR_50(input_size, embedding_size),
                     'IR_101': IR_101(input_size, embedding_size),
                     'IR_152': IR_152(input_size, embedding_size),
                     'IR_SE_50': IR_SE_50(input_size, embedding_size),
                     'IR_SE_101': IR_SE_101(input_size, embedding_size),
                     'IR_SE_152': IR_SE_152(input_size, embedding_size),
                     'Baseline': Encoder()}

    NUM_CLASS_DICT = {'AGE': 9,
                      'RACE': 7,
                      'GENDER': 2}

    if DATASET == 'CelebA':
        NUM_CLASS = 2
    else:
        NUM_CLASS = NUM_CLASS_DICT[main_task]

    embedding_net = BACKBONE_DICT[backbone].to(DEVICE)

    HEAD_DICT = {'ArcFace': ArcFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=GPU_ID),
                 'CosFace': CosFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Linear': Softmax(in_features=embedding_size, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Baseline': Public_Classifier(output=NUM_CLASS).to(DEVICE)
                 }

    head = HEAD_DICT[head_name].to(DEVICE)

    print("embed path: ", embed_path)

    if checkpoint:
        print("Checkpoint!")
        checkpoint = torch.load(embed_path, map_location=DEVICE, weights_only=False)
        state_dict_embed = checkpoint['backbone_state_dict']
        head_state_dict = checkpoint['head_state_dict']
    else:
        print("No Checkpoint!")
        state_dict_embed = torch.load(embed_path, map_location=DEVICE)
        head_state_dict = torch.load(head_path, map_location=DEVICE)

    load_model(embedding_net, state_dict_embed)
    load_model(head, head_state_dict)

    if finetuning:
        siamese_finetuning(train_loader, embedding_net, head, DATASET, device, main_task, epochs_fine_tuning, lr)

    if args.pca:
        features = extract_features(copy.deepcopy(embedding_net).to(DEVICE), train_loader, DATASET, device=device)
        pca_model = fit_pca(features, output_dim=8)

        input_dim = pca_model.components_.shape[1]
        output_dim = pca_model.components_.shape[0]
        weight = torch.tensor(pca_model.components_.T, dtype=torch.float32).to(DEVICE)

        pca = PCAReduction(input_dim=input_dim, output_dim=output_dim, pretrained_matrix=weight).to(DEVICE)
        pca_decom = PCADecompression(weight).to(DEVICE)

    noise = NoiseInjection(scale=args.scale,device=DEVICE).to(DEVICE)

    if args.pca:
        pca_noise_module = PrivacyEmbedding(pca, noise)

        embed_pca = nn.Sequential(embedding_net, pca_noise_module)
        embed_pca_decomp = copy.deepcopy(nn.Sequential(embed_pca, pca_decom)).to(DEVICE)
    else:
        embed_pca_decomp = nn.Sequential(embedding_net,noise)

    n_trials_adv = n_trials
    min_budget_adv = 10
    max_budget_adv = adv_epochs
    eta_adv = 3

    torch.cuda.empty_cache()

    if args.adv_config_path is not None:
        with open(args.adv_config_path, 'rb') as f:
            adv_config = pickle.load(f)
    elif args.hpo:
        adv_config = get_best_adv_config(embed_pca_decomp, sensitive_task, head_name, n_trials_adv, min_budget_adv,
                                     max_budget_adv, eta_adv, embedding_size, DATA_ROOT, DATASET)
    else:
        cs = ConfigurationSpace()

        # HyperParameters to optimize
        lr_adam = Float("lr_adam", (1e-5, 1e-2), default=1e-3, log=True)
        lr_sgd = Float("lr_sgd", (0.0001, 1), log=True, default=0.1)
        arch = Categorical("head", ["Baseline", "CosFace", "ArcFace", "Linear", "5Layer"], default="Baseline")
        loss = Categorical("loss", ["Focal", "CrossEntropy"], default="Focal")
        optimizer = Categorical("optimizer", ["Adam", "SGD"], default="SGD")
        batch_size = Integer("batch_size", (30, 70), default=50)

        # Conditions
        use_lr_adam = InCondition(lr_adam, parent=optimizer, values=['Adam'])
        use_lr_sgd = InCondition(lr_sgd, parent=optimizer, values=['SGD'])

        cs.add_hyperparameters([
            lr_adam,
            lr_sgd,
            arch,
            loss,
            optimizer,
            batch_size,
        ])
        cs.add_conditions([use_lr_adam, use_lr_sgd])

        adv_config = Configuration(cs, values={'batch_size': 31,
                                               'head': 'Baseline',
                                               'loss': 'CrossEntropy',
                                               'optimizer': 'SGD',
                                               'lr_sgd': 0.0515644226134,
                                               })

    print(adv_config)

    epoch_val_acc = validation_bio(val_loader, DEVICE, embed_pca_decomp, main_task, head, DATASET)

    print("Val Accuracy: ", (epoch_val_acc.avg / 100))
    print("Arch, main, sensitive: ", backbone, main_task, sensitive_task)

    privacy_risk = aux_adversary_privacy_risk(embed_pca_decomp, embedding_size, DEVICE, sensitive_task,
                                              adv_config, adv_epochs, DATASET)

    print("Val Accuracy: ", (epoch_val_acc.avg / 100))
    print("Privacy Risk: ", privacy_risk)

    wandb.log({"Val Accuracy": (epoch_val_acc.avg / 100),
               "Privacy Risk": privacy_risk})


if __name__ == "__main__":
    args = parse_args()
    main(args)
