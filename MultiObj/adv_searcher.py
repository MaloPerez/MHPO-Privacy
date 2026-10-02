"""Searches for the adversary hyperparameters used to measure privacy risk (PR)
during an M-HPO run (get_best_adv_config)."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import warnings
import functools
import pandas as pd
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from callback import CustomCallback_Adv
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
from smac.runner.target_function_runner import TargetFunctionRunner

import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

import numpy as np
from data_loaders import FairFaceDataset, load_celebA_data
from config_single import configurations
from baselines import Encoder, Public_Classifier
import argparse
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from metrics import ArcFace, CosFace, Softmax
from focal import FocalLoss
from MAC_model import BioTask_2Layers, BioTask_1Layer, BioTask_5Layers

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import re


def train_adv(config, seed, budget, embed_net, sensitive_task, DATA_ROOT, transforms_aux,
              EMBEDDING_SIZE, DATASET, disantangled=False):
    NUM_EPOCH = int(budget)
    HEAD_NAME = config["head"]
    LOSS_NAME = config["loss"]
    OPTIMIZER_NAME = config["optimizer"]
    BATCH_SIZE = config["batch_size"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    GPU_ID = [0]

    if DATASET == 'CelebA':
        train_dataset, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", transforms_aux)
        val_dataset, _, weights_val = load_celebA_data(DATA_ROOT, "valid", "attr", transforms_aux)

        WEIGHTS_SENSITIVE = weights_train[sensitive_task].to(device)
    else:
        CSV_TRAIN_PATH = f'{_ROOT}/data/FairFace/fairface_label_train.csv'
        CSV_VAL_PATH = f'{_ROOT}/data/FairFace/fairface_label_val.csv'
        train_dataset = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, transforms_aux)
        val_dataset = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, transforms_aux)

    INPUT_SIZE = [112, 112]
    sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True,
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

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 '5Layer': BioTask_5Layers(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS).to(device),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Linear': Softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Baseline': Public_Classifier(input=EMBEDDING_SIZE, output=NUM_CLASS).to(device)
                 }

    adv_model = HEAD_DICT[HEAD_NAME]

    if DATASET == 'CelebA':
        LOSS_DICT = {'Focal': FocalLoss(weight=WEIGHTS_SENSITIVE).to(device),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS_SENSITIVE).to(device)}
    else:
        _sens_weight = train_dataset.class_weights[CLASS_WEIGHT_IDX[sensitive_task]].to(device)
        LOSS_DICT = {'Focal': FocalLoss(weight=_sens_weight),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=_sens_weight)}
    loss_crit_adv = LOSS_DICT[LOSS_NAME]

    if OPTIMIZER_NAME == 'SGD':
        LR = config["lr_sgd"]
        optim_adv = optim.SGD(adv_model.parameters(), lr=LR)

    elif OPTIMIZER_NAME == "Adam":
        LR = config["lr_adam"]
        optim_adv = optim.Adam(adv_model.parameters(), lr=LR)

    # embed_net is frozen (only ever called under torch.no_grad() below) -- eval()
    # keeps its BatchNorm layers on running stats instead of batch stats, which
    # also avoids a crash ("Expected more than 1 value per channel when training")
    # whenever drop_last=False leaves a size-1 tail batch for some batch_size choice.
    embed_net.eval()

    for e in range(NUM_EPOCH):
        num_batches = len(train_loader)

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
                    input_adv = embed_net(imgs)[0].to(device)
                else:
                    input_adv = embed_net(imgs).to(device)

            if isinstance(adv_model, ArcFace) or isinstance(adv_model, CosFace):
                forward_adv = adv_model.forward(input_adv, target).to(device)
            else:
                forward_adv = adv_model.forward(input_adv).to(device)

            # forward_adv = adv_model.forward(input_adv).to(device)
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
                input_adv = embed_net(imgs)[0].to(device)
            else:
                input_adv = embed_net(imgs).to(device)

        if isinstance(adv_model, CosFace) or isinstance(adv_model, ArcFace):
            forward_adv = adv_model(input_adv, target).to(device)
        else:
            forward_adv = adv_model(input_adv).to(device)

        all_preds = all_preds + forward_adv.topk(1)[1].cpu().tolist()
        all_targets = all_targets + target.tolist()

    bal_acc_adv = balanced_accuracy_score(all_targets, all_preds)

    costs = {
        "1 - adv_acc": 1 - bal_acc_adv,
    }

    return costs


def get_best_adv_config(embed_net, sensitive_task, arch_name, n_trials, min_budget, max_budget, eta,
                        EMBEDDING_SIZE, DATA_ROOT, DATASET, disantangled=False):
    cs = ConfigurationSpace()

    # HyperParameters to optimize
    lr_adam = Float("lr_adam", (1e-5, 1e-2), default=1e-3, log=True)
    lr_sgd = Float("lr_sgd", (0.0001, 1), log=True, default=0.1)
    if arch_name == 'Baseline':
        arch = Categorical("head", ["Baseline"], default="Baseline")
    else:
        arch = Categorical("head", ["CosFace", "ArcFace", "Linear", "5Layer"], default="CosFace")
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

    objectives = ["1 - adv_acc"]

    transforms_aux = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),  # smaller side resized
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    scenario = Scenario(
        cs,
        walltime_limit=400000000000,  # After x seconds, we stop the hyperparameter optimization
        n_trials=n_trials,
        objectives=objectives,
        min_budget=min_budget,
        max_budget=max_budget,
        n_workers=3
    )

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=3)
    intensifier = MultiFidelityFacade.get_intensifier(scenario, eta=eta, incumbent_selection="highest_budget")

    cost_function = functools.partial(train_adv, embed_net=embed_net, sensitive_task=sensitive_task,
                                      DATA_ROOT=DATA_ROOT, transforms_aux=transforms_aux,
                                      EMBEDDING_SIZE=EMBEDDING_SIZE, DATASET=DATASET, disantangled=disantangled)

    smac = MultiFidelityFacade(
        scenario,
        cost_function,
        initial_design=initial_design,
        overwrite=True,
        callbacks=[CustomCallback_Adv()],
        intensifier=intensifier
    )

    incumbent = smac.optimize()

    incumbent_df = pd.DataFrame(columns=[param.name for param in cs.get_hyperparameters()] + objectives + ['hash'])

    cost = smac.runhistory.average_cost(incumbent)
    print("Incumbent", dict(incumbent))
    print("Cost", cost)
    values = dict(incumbent)
    for i, obj in enumerate(objectives):
        values[obj] = cost
    values['hash'] = get_config_hash(incumbent)
    incumbent_df = pd.concat(
        [incumbent_df, pd.DataFrame([values])],
        ignore_index=True
    )

    print(incumbent_df)

    return incumbent
