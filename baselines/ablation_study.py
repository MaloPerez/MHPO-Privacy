"""Ablation study validating the privacy-risk adversary construction: main-task
accuracy and privacy risk measured at different training epochs and across
adversaries trained with random hyperparameter configurations (paper Fig. 2)."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import math
import hashlib
import io
import copy
import itertools
from sklearn.metrics import balanced_accuracy_score

import sys

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
import sys

from smac.multi_objective.parego import ParEGO
from smac.utils.configspace import get_config_hash

import os

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
from MAC_model import BioTask_2Layers, BioTask_1Layer, BioTask_5Layers

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Trains MAC model on selected public dataset')
    parser.add_argument('--batch-size', type=int, default='50',
                        help='balanced_ce or focal')
    parser.add_argument('--type-init', type=str, default="normal",
                        help='type of initialization for weights of mac model (default: normal)')
    parser.add_argument('--data-path', type=str, default=f'{_ROOT}/MAC/data',
                        help='path to the directory containing the data (default: deep-privacy/data)')
    parser.add_argument('--head-path', type=str)
    parser.add_argument('--embed-path-1', type=str)
    parser.add_argument('--embed-path-2', type=str)
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


def hash_model_state_dict(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    hash_val = hashlib.sha256(buffer.read()).hexdigest()
    return hash_val


def init_weights_default(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)


def aux_adversary_privacy_risk(embedding_net, embedding_size, device, sensitive_task, adv_config, adv_epochs,
                               DATASET, disantangled=False):
    adv_batch_size = adv_config["batch_size"]
    adv_name = adv_config['head']
    adv_loss_name = adv_config['loss']
    adv_optim_name = adv_config['optimizer']

    print(adv_config)
    print(hash_model_state_dict(embedding_net))

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

    print(adv_model)

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
        print('LR: ', LR)
        optim_adv = optim.SGD(adv_model.parameters(), lr=LR)

    elif adv_optim_name == "Adam":
        LR = adv_config["lr_adam"]
        print('LR: ', LR)
        optim_adv = optim.Adam(adv_model.parameters(), lr=LR)

    print("after init")
    for name, param in adv_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    for e in tqdm(range(adv_epochs)):
        print("Epoch ", e)

        for i, batch in enumerate(train_loader):
            adv_model.train()
            optim_adv.zero_grad()

            if DATASET == 'CelebA':
                imgs = batch[0].to(device)
                target = batch[1][:, sensitive_task].reshape((-1)).to(device)
            else:
                imgs = batch['img'].to(device)
                target = batch[sensitive_task].reshape((-1)).to(device)

            with torch.no_grad():
                input_adv = embedding_net(imgs).to(device)

            if i == 0:
                print(input_adv)

            if isinstance(adv_model, ArcFace) or isinstance(adv_model, CosFace):
                forward_adv = adv_model.forward(input_adv, target).to(device)
            else:
                forward_adv = adv_model.forward(input_adv).to(device)

            loss_adv = loss_crit_adv.forward(forward_adv, target)

            wandb.log({"Val Accuracy": loss_adv})

            loss_adv.backward()

            if i == 0:
                for name, param in adv_model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad mean: {param.grad.abs().mean().item()}")
                    else:
                        print(f"{name} has no gradient!")

            optim_adv.step()

    all_preds = []
    all_targets = []

    for _, batch in enumerate(val_loader):
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

    print(bal_acc_adv)

    return bal_acc_adv


def main():
    wandb.init(project="Ablation Study")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [0]

    # loading args
    batch_size = args.batch_size
    embed_path_1 = args.embed_path_1
    embed_path_2 = args.embed_path_2
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

    embedding_net_1 = BACKBONE_DICT[backbone].to(DEVICE)
    embedding_net_2 = copy.deepcopy(embedding_net_1).to(DEVICE)

    if checkpoint:
        print("Checkpoint!")
        checkpoint_1 = torch.load(embed_path_1, map_location=DEVICE)
        state_dict_embed_1 = checkpoint_1['backbone_state_dict']

        checkpoint_2 = torch.load(embed_path_2, map_location=DEVICE)
        state_dict_embed_2 = checkpoint_2['backbone_state_dict']
    else:
        print("No Checkpoint!")
        state_dict_embed_1 = torch.load(embed_path_1, map_location=DEVICE)
        state_dict_embed_2 = torch.load(embed_path_2, map_location=DEVICE)

    load_model(embedding_net_1, state_dict_embed_1)
    load_model(embedding_net_2, state_dict_embed_2)

    n_trials_adv = n_trials
    min_budget_adv = 10
    max_budget_adv = adv_epochs
    eta_adv = 3

    if args.adv_config_path is not None:
        with open(args.adv_config_path, 'rb') as f:
            adv_config_1 = pickle.load(f)
            adv_config_2 = adv_config_1
    else:
        adv_config_1 = get_best_adv_config(embedding_net_1, sensitive_task, head_name, n_trials_adv, min_budget_adv,
                                           max_budget_adv, eta_adv, embedding_size, DATA_ROOT, DATASET)

        adv_config_2 = get_best_adv_config(embedding_net_2, sensitive_task, head_name, n_trials_adv, min_budget_adv,
                                           max_budget_adv, eta_adv, embedding_size, DATA_ROOT, DATASET)

    print(adv_config_1)
    print(adv_config_2)

    print("Everything: ", embedding_net_1, embedding_size, DEVICE, sensitive_task, adv_config_1, adv_epochs, DATASET)

    privacy_risk_embed_1_config_1 = aux_adversary_privacy_risk(embedding_net_1, embedding_size, DEVICE, sensitive_task,
                                                               adv_config_1, adv_epochs, DATASET)
    print("Embed 1 Config 1: ", privacy_risk_embed_1_config_1)
    privacy_risk_embed_1_config_2 = aux_adversary_privacy_risk(embedding_net_1, embedding_size, DEVICE, sensitive_task,
                                                               adv_config_2, adv_epochs, DATASET)
    print("Embed 1 Config 2: ", privacy_risk_embed_1_config_2)
    privacy_risk_embed_2_config_1 = aux_adversary_privacy_risk(embedding_net_2, embedding_size, DEVICE, sensitive_task,
                                                               adv_config_1, adv_epochs, DATASET)
    print("Embed 2 Config 1: ", privacy_risk_embed_2_config_1)
    privacy_risk_embed_2_config_2 = aux_adversary_privacy_risk(embedding_net_2, embedding_size, DEVICE, sensitive_task,
                                                               adv_config_2, adv_epochs, DATASET)
    print("Embed 2 Config 2: ", privacy_risk_embed_2_config_2)


    wandb.log({"Embed 1 Config 1: ": privacy_risk_embed_1_config_1,
               "Embed 1 Config 2: ": privacy_risk_embed_1_config_2,
               "Embed 2 Config 1: ": privacy_risk_embed_2_config_1,
               "Embed 2 Config 2: ": privacy_risk_embed_2_config_2})


if __name__ == "__main__":
    args = parse_args()
    main()
