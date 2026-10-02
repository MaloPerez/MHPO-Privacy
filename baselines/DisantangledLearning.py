"""Disentangled Representation Learning (DL) privacy-defense baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import math
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
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from baselines import Encoder, Public_Classifier
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
    parser.add_argument('--var', type=float, default=1)
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=False)
    parser.add_argument('--lr', type=float, default=0.9)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--no-noise', dest='noise', action='store_false')
    parser.add_argument('--dataset', type=str, default='FairFace')
    parser.add_argument('--epochs-auto-enc', type=int, default='50')
    parser.add_argument('--n-trials', type=int, default='20')
    args = parser.parse_args()
    return args


def load_model(model, state_dict):
    model.load_state_dict(state_dict)
    return model


def validation_bio(val_loader, DEVICE, BACKBONE, MAIN_TASK, HEAD):
    batch = 0
    top1 = AverageMeter()

    for batch_sample in tqdm(iter(val_loader)):

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
        LOSS_DICT = {'Focal': FocalLoss(),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS_SENSITIVE).to(device)}
    else:
        LOSS_DICT = {'Focal': FocalLoss(),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=train_dataset
                                                         .class_weights[CLASS_WEIGHT_IDX[sensitive_task]].to(device))}
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

            if isinstance(adv_model, ArcFace) or isinstance(adv_model, CosFace):
                forward_adv = adv_model.forward(input_adv, target).to(device)
            else:
                forward_adv = adv_model.forward(input_adv).to(device)

            loss_adv = loss_crit_adv.forward(forward_adv, target)

            loss_adv.backward()

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


def moment_loss(z_1, z_2, alpha=2):
    moment_1 = torch.mean(z_1 ** alpha, dim=0)
    moment_2 = torch.mean(z_2 ** alpha, dim=0)
    return torch.norm(moment_1 - moment_2, p=2)


def feature_disentanglement_loss(z_main_1, z_main_2, lambda_alpha, alpha_max=2):
    loss = sum(moment_loss(z_main_1, z_main_2, alpha) * lambda_alpha for alpha in range(1, alpha_max + 1))
    return loss


def gender_discrimination_loss(z_priv_1, z_priv_2, lambda_beta, beta_max=2, sigma=1.0):
    loss = sum(torch.exp(-moment_loss(z_priv_1, z_priv_2, beta) / (2 * sigma ** 2)) * lambda_beta for beta in
               range(1, beta_max + 1))
    return loss


def pair_inputs(batch_sample, dataset, device, priv_class):
    if dataset == 'CelebA':
        inputs = batch_sample[0].to(device)
        labels = batch_sample[1][:, priv_class].reshape((-1)).to(device)
    else:
        inputs = batch_sample['img'].to(device)
        labels = batch_sample[priv_class.lower()].reshape((-1)).to(device)

    classes = torch.unique(labels).tolist()
    pairs = list(itertools.combinations(classes, r=2))

    loaders = []

    for pair in pairs:
        idx_1 = (labels == pair[0]).nonzero().flatten()
        idx_2 = (labels == pair[1]).nonzero().flatten()

        inputs_1 = inputs[idx_1, :]
        inputs_2 = inputs[idx_2, :]

        labels_1 = labels[idx_1]
        labels_2 = labels[idx_2]

        loaders.append([[inputs_1, labels_1], [inputs_2, labels_2]])

    return loaders


def train_PFRNET(model, train_loader, dataset, embed_net, device, priv_class, epochs):
    reconstruction_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    embed_net.eval()

    # Use train loader to get some data.
    total_epochs = epochs
    for epoch in range(0, total_epochs):
        print('Epoch:{}/{}', epoch + 1, total_epochs)
        for batch_sample in tqdm(iter(train_loader)):

            loaders = pair_inputs(batch_sample, dataset, device, priv_class)

            # Compute loss
            lambda_dis, lambda_disc, sigma = 0.01, 0.01, 1 / math.sqrt(2)
            loss_recon = 0
            loss_disentangle = 0
            loss_discriminate = 0

            for pairs in loaders:
                # Forward pass
                input_1 = pairs[0][0].to(device)
                input_2 = pairs[1][0].to(device)

                # embed_net is frozen -- only model.parameters() (PFRNet) get
                # optimizer.step() below -- so tracking gradients through the
                # backbone here is pure wasted backward-pass compute, done twice
                # per batch (once per paired input). no_grad() skips it entirely.
                with torch.no_grad():
                    input_1 = embed_net(input_1)
                    input_2 = embed_net(input_2)

                z_main_1, z_priv_1, x_recon_1 = model(input_1)
                z_main_2, z_priv_2, x_recon_2 = model(input_2)

                loss_recon = loss_recon + reconstruction_loss(x_recon_1.to(device), input_1.to(device)) \
                             + reconstruction_loss(x_recon_2.to(device), input_2.to(device))
                loss_disentangle = loss_disentangle + feature_disentanglement_loss(z_main_1.to(device),
                                                                                   z_main_2.to(device),
                                                                                   lambda_dis)
                loss_discriminate = loss_discriminate + gender_discrimination_loss(z_priv_1.to(device),
                                                                                   z_priv_2.to(device),
                                                                                   lambda_disc,
                                                                                   sigma=sigma)

            # .item() detaches from the autograd graph before logging, so wandb doesn't
            # pin these tensors' computation graphs in memory.
            wandb.log({'Loss Recon': loss_recon.item(),
                       'Loss Disentangle': loss_disentangle.item(),
                       'Loss Discriminate': loss_discriminate.item()}
                      )

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss = loss + loss_recon + lambda_dis * loss_disentangle + lambda_disc * loss_discriminate

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main(args):
    wandb.init(project="Disantangled Learning")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [0]

    # loading args
    batch_size = args.batch_size
    embed_path = args.embed_path
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
    epochs_auto_enc = args.epochs_auto_enc
    n_trials = args.n_trials

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

    print("embed path: ", embed_path)

    if checkpoint:
        print("Checkpoint!")
        checkpoint = torch.load(embed_path, map_location=DEVICE)
        state_dict_embed = checkpoint['backbone_state_dict']
    else:
        print("No Checkpoint!")
        state_dict_embed = torch.load(embed_path, map_location=DEVICE)

    load_model(embedding_net, state_dict_embed)

    disantangler = PFRNet(input_dim=embedding_size, id_latent_dim=int(0.95 * embedding_size),
                          attr_latent_dim=embedding_size - int(0.95 * embedding_size)).to(DEVICE)

    train_PFRNET(disantangler, train_loader, DATASET, embedding_net, DEVICE, sensitive_task, epochs_auto_enc)

    n_trials_adv = n_trials
    min_budget_adv = 10
    max_budget_adv = adv_epochs
    eta_adv = 3

    embed_disantangle = nn.Sequential(embedding_net, disantangler)

    if args.adv_config_path is not None:
        with open(args.adv_config_path, 'rb') as f:
            main_config = pickle.load(f)
    else:
        main_config = get_best_adv_config(embed_disantangle, main_task.lower(), head_name, n_trials_adv, min_budget_adv,
                                      max_budget_adv, eta_adv, int(0.95 * embedding_size), DATA_ROOT, DATASET, disantangled=True)

    print(main_config)

    if args.adv_config_path is not None:
        adv_config = main_config
    else:
        adv_config = get_best_adv_config(embed_disantangle, sensitive_task, head_name, n_trials_adv, min_budget_adv,
                                     max_budget_adv, eta_adv, int(0.95 * embedding_size), DATA_ROOT, DATASET, disantangled=True)

    print(adv_config)

    if DATASET == 'FairFace':
        main_task = main_task.lower()

    epoch_val_acc = aux_adversary_privacy_risk(embed_disantangle, int(0.95 * embedding_size), DEVICE, main_task,
                                               main_config, adv_epochs, DATASET, disantangled=True)
    privacy_risk = aux_adversary_privacy_risk(embed_disantangle, int(0.95 * embedding_size), DEVICE, sensitive_task,
                                              adv_config, adv_epochs, DATASET, disantangled=True)

    print("Val Accuracy: ", epoch_val_acc)
    print("Privacy Risk: ", privacy_risk)

    wandb.log({"Val Accuracy": epoch_val_acc,
               "Privacy Risk": privacy_risk})


if __name__ == "__main__":
    args = parse_args()
    main(args)
