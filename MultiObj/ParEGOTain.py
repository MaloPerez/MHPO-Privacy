"""Multi-objective (accuracy + privacy), multi-fidelity SMAC/ParEGO HPO search --
the paper's core M-HPO method for training privacy-preserving embedding networks."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import warnings
import functools
import pandas as pd
import csv
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from callback import CustomCallback
from adv_searcher import get_best_adv_config
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
from smac.runner.target_function_runner import TargetFunctionRunner

import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC', 'MultiObj']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

import numpy as np
from data_loaders import FairFaceDataset, load_celebA_data
from baselines import Encoder, Public_Classifier
from config_multi import configurations
import argparse
from model_resnet import ResNet_50, ResNet_101, ResNet_152
from model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from metrics import ArcFace, CosFace, Softmax
from focal import FocalLoss
from focal import FocalLoss
from utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from MAC_model import BioTask_2Layers, BioTask_1Layer, BioTask_5Layers

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import re


def parse_args():
    parser = argparse.ArgumentParser(description='Get best hyperparameters for accuracy and ')
    parser.add_argument('--init-configs', type=int, default=5)
    parser.add_argument('--min-budget', type=int, default=1)
    parser.add_argument('--max-budget', type=int, default=120)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--eta', type=int, default=2)
    parser.add_argument('--exp-path', type=str, default='test_multi')
    parser.add_argument('--checkpointing', type=bool, default=True)
    parser.add_argument('--config-file', type=int, default=1)
    args = parser.parse_args()
    return args


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def choose_gpu(device_list):
    mem = []
    for i in device_list:
        r = torch.cuda.memory_reserved(i)
        a = torch.cuda.memory_allocated(i)
        f = r - a  # free inside reserved
        mem.append(f)
    print(mem)
    best_idx = mem.index(min(mem))
    return best_idx


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

        if isinstance(HEAD, CosFace) or isinstance(HEAD, ArcFace):
            outputs = HEAD(features, labels)
        else:
            outputs = HEAD(features)

        # measure val accuracy and val loss
        prec1 = accuracy(outputs.data, labels)
        top1.update(prec1[0].item(), current_batch_size)

        batch += 1  # batch index

    epoch_val_acc = top1

    return epoch_val_acc


def aux_adversary_privacy_risk(embedding_net, embedding_size, device, sensitive_task, adv_config, DATASET):
    adv_batch_size = adv_config["batch_size"]
    adv_name = adv_config['head']
    adv_loss_name = adv_config['loss']
    adv_optim_name = adv_config['optimizer']

    adv_epochs = 50

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

    INPUT_SIZE = [112, 112]
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
        'Baseline': Public_Classifier(output=NUM_CLASS).to(device),
        'ArcFace': ArcFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
        'CosFace': CosFace(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
        'Linear': Softmax(in_features=embedding_size, out_features=NUM_CLASS, device_id=[0]),
    }

    adv_model = ADV_DICT[adv_name]

    if DATASET == 'CelebA':
        LOSS_DICT = {'Focal': FocalLoss(weight=WEIGHTS_SENSITIVE.to(device)),
                     'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS_SENSITIVE.to(device))}
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
                input_adv = embedding_net(imgs).to(device)

            if isinstance(adv_model, CosFace) or isinstance(adv_model, ArcFace):
                forward_adv = adv_model(input_adv, target).to(device)
            else:
                forward_adv = adv_model(input_adv).to(device)

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
            input_adv = embedding_net(imgs).to(device)

        if isinstance(adv_model, CosFace) or isinstance(adv_model, ArcFace):
            forward_adv = adv_model(input_adv, target).to(device)
        else:
            forward_adv = adv_model(input_adv).to(device)

        all_preds = all_preds + forward_adv.topk(1)[1].cpu().tolist()
        all_targets = all_targets + target.tolist()

    bal_acc_adv = balanced_accuracy_score(all_targets, all_preds)

    return bal_acc_adv


def plot_pareto(smac, incumbents):
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    fig = plt.gcf()

    wandb.log({"Pareto": fig})


def train_bio_task(config, seed, budget, model_path, adv_config_path, checkpointing, config_file, adv_search_mode=False):
    if adv_search_mode and os.path.isdir(adv_config_path):
        return

    cfg_file = configurations[config_file]

    DATASET = cfg_file['DATASET']
    DATA_ROOT = cfg_file['DATA_ROOT']  # the parent root where your train/val/test data are stored
    CSV_TRAIN_PATH = cfg_file['CSV_TRAIN_PATH']
    CSV_VAL_PATH = cfg_file['CSV_VAL_PATH']

    MAIN_TASK = cfg_file['MAIN_TASK']
    SENSITIVE_TASKS = cfg_file['SENSITIVE_TASK'].copy()
    print(SENSITIVE_TASKS)
    NUM_CLASS_DICT = {'AGE': 9,
                      'RACE': 7,
                      'GENDER': 2}

    if DATASET == 'CelebA':
        NUM_CLASS = 2
    else:
        NUM_CLASS = NUM_CLASS_DICT[MAIN_TASK]

    INPUT_SIZE = cfg_file['INPUT_SIZE']
    EMBEDDING_SIZE = cfg_file['EMBEDDING_SIZE']
    BACKBONE_NAME = cfg_file['BACKBONE_NAME']
    ADV_NAME = cfg_file['ADV_NAME']
    RGB_MEAN = cfg_file['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg_file['RGB_STD']
    DROP_LAST = cfg_file['DROP_LAST']
    STAGES = cfg_file['STAGES']
    MOMENTUM = cfg_file['MOMENTUM']

    MULTI_GPU = cfg_file['MULTI_GPU']
    GPU_ID = cfg_file['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg_file['PIN_MEMORY']
    NUM_WORKERS = cfg_file['NUM_WORKERS']

    if MULTI_GPU:
        GPU_ID = choose_gpu(GPU_ID)
        device = GPU_ID
    else:
        GPU_ID = GPU_ID[0]
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in
        transforms.ToPILImage(),
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN,
                             std=RGB_STD),
    ])

    WEIGHTS = None
    SENSITIVE_TASKS_NAMES = SENSITIVE_TASKS.copy()

    if DATASET == 'CelebA':
        print("SENSTIVE TASKS", SENSITIVE_TASKS)
        dataset_train, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", train_transform)
        dataset_val, _, weights_val = load_celebA_data(DATA_ROOT, "valid", "attr", train_transform)

        INDEX_LABEL = dict(zip(attr, range(len(attr))))
        MAIN_TASK = INDEX_LABEL[MAIN_TASK]
        WEIGHTS = weights_train[MAIN_TASK].to(device)
        for i in range(len(SENSITIVE_TASKS)):
            SENSITIVE_TASKS[i] = INDEX_LABEL[SENSITIVE_TASKS_NAMES[i]]
        print("SENSITIVE TASKS NAMES 1:", SENSITIVE_TASKS_NAMES)
        # WEIGHTS_SENSITIVE = weights_train[SENSITIVE_TASKS[0]].to(device)
    else:
        dataset_train = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, train_transform)
        dataset_val = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, train_transform)

    NUM_EPOCH = int(budget)
    HEAD_NAME = config["head"]
    LOSS_NAME = config["loss"]
    OPTIMIZER_NAME = config["optimizer"]
    BATCH_SIZE = config["batch_size"]
    WEIGHT_DECAY = config['weight_decay']

    sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(BATCH_SIZE / 2), sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )

    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_101': ResNet_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_152': ResNet_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_50': IR_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_101': IR_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_152': IR_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_101': IR_SE_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_152': IR_SE_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'Baseline': Encoder()}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    # print("{} Backbone Generated".format(BACKBONE_NAME))
    # print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=[GPU_ID]),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=[GPU_ID]),
                 'Linear': Softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=[GPU_ID]),
                 'Baseline': Public_Classifier(output=NUM_CLASS).to(device)
                 }

    HEAD = HEAD_DICT[HEAD_NAME]
    # print("=" * 60)
    # print("{} Head Generated".format(HEAD_NAME))
    # print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(weight=WEIGHTS),
                 'CrossEntropy': nn.CrossEntropyLoss(weight=WEIGHTS)}
    LOSS = LOSS_DICT[LOSS_NAME]
    # print("=" * 60)
    # print("{} Loss Generated".format(LOSS_NAME))
    # print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)  # separate batch_norm
        # parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        if HEAD:
            _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)  # separate batch_norm
        # parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        if HEAD:
            _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

    LR = None

    if OPTIMIZER_NAME == 'SGD':
        LR = config["lr_sgd"]
        OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                               {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
    elif OPTIMIZER_NAME == "Adam":
        LR = config["lr_adam"]
        OPTIMIZER = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                {'params': backbone_paras_only_bn}], lr=LR)



    config_hash = get_config_hash(config)
    config_model_path = os.path.join(model_path, config_hash)

    # optionally resume from a checkpoint
    start_epoch = 0

    if os.path.exists(config_model_path) and checkpointing:
        print("=" * 60)
        print(sorted(os.listdir(config_model_path), key=numericalSort))
        checkpoint_name = list(filter(lambda k: 'checkpoint' in k, sorted(os.listdir(config_model_path),
                                                                          key=numericalSort)))[-1]
        checkpoint_path = os.path.join(config_model_path, checkpoint_name)
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint['epoch'])
        if checkpoint['epoch'] < NUM_EPOCH:
            print("Loading Config {} Epoch '{}'".format(config_hash, checkpoint['epoch']))
            BACKBONE.load_state_dict(checkpoint['backbone_state_dict'])
            HEAD.load_state_dict(checkpoint['head_state_dict'])
            start_epoch = int(checkpoint['epoch'])
        if checkpoint['epoch'] == NUM_EPOCH:
            BACKBONE.load_state_dict(checkpoint['backbone_state_dict'])
            HEAD.load_state_dict(checkpoint['head_state_dict'])
            costs = {
                "1 - main_task_accuracy": checkpoint["1 - main_task_accuracy"],
            }
            for task in SENSITIVE_TASKS_NAMES:
                costs['privacy_risk_{}'.format(task)] = checkpoint['privacy_risk_{}'.format(task)]

            return costs


    print("Config Used : ", config)
    print("=" * 60)

    BACKBONE = BACKBONE.to(device)

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    last_loss = 0
    last_acc = 0

    for epoch in range(start_epoch + 1, NUM_EPOCH):  # start training process

        if epoch == STAGES[0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR
            # manually (with slight modification) once plateau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        if HEAD:
            HEAD.train()

        losses = AverageMeter()
        top1 = AverageMeter()

        for batch_sample in tqdm(iter(train_loader)):

            if DATASET == 'CelebA':
                inputs = batch_sample[0].to(device)
                labels = batch_sample[1][:, MAIN_TASK].reshape((-1)).to(device)
            else:
                inputs = batch_sample['img'].to(device)
                labels = batch_sample[MAIN_TASK.lower()].reshape((-1)).to(device)

            current_batch_size = inputs.size(0)

            features = BACKBONE(inputs).to(device)

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                # compute output

            if isinstance(HEAD, CosFace) or isinstance(HEAD, ArcFace):
                outputs = HEAD(features, labels).to(device)
            else:
                outputs = HEAD(features).to(device)

            batch_loss = LOSS(outputs, labels)

            prec1 = accuracy(outputs.data, labels)
            top1.update(prec1[0].item(), current_batch_size)

            losses.update(batch_loss.data.item(), current_batch_size)

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            batch_loss.backward()
            OPTIMIZER.step()

            batch += 1  # batch index

        last_loss = losses.avg
        last_acc = top1.avg

        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch + 1, NUM_EPOCH, loss=losses,
                                                                       top1=top1))

        print("=" * 60)

    print("SENSITIVE TASKS NAMES 2:", SENSITIVE_TASKS_NAMES)

    epoch_val_acc = validation_bio(val_loader, device, BACKBONE, MAIN_TASK, HEAD, DATASET)
    privacy_risks = []

    adv_config_file_name = os.path.join(adv_config_path, 'adv_config.pkl')

    if adv_search_mode and (not os.path.isdir(adv_config_path)) and len(SENSITIVE_TASKS) == 1:
        n_trials_adv = 20
        min_budget_adv = 10
        max_budget_adv = 50
        eta_adv = 3
        adv_config = get_best_adv_config(BACKBONE, SENSITIVE_TASKS[0], ADV_NAME, n_trials_adv, min_budget_adv,
                                         max_budget_adv, eta_adv, EMBEDDING_SIZE, DATA_ROOT, DATASET)
        os.makedirs(adv_config_path)
        with open(os.path.join(adv_config_path, 'adv_config.pkl'), 'wb') as f:
            pickle.dump(adv_config, f)
    elif os.path.isfile(adv_config_file_name) and len(SENSITIVE_TASKS) == 1:
        with open(adv_config_file_name, 'rb') as f:
            adv_config = pickle.load(f)
    elif len(SENSITIVE_TASKS) == 1:
        raise "Adversary not found yet"
    else:
        raise "Not implemented yet"

    print("SENSITIVE TASKS NAMES 3:", SENSITIVE_TASKS_NAMES)
    for task in SENSITIVE_TASKS:
        privacy_risk = aux_adversary_privacy_risk(BACKBONE, EMBEDDING_SIZE, device, task, adv_config, DATASET)
        privacy_risks.append(privacy_risk)

    print("Privacy risks", privacy_risks)

    print("COSTS: ", 1 - (epoch_val_acc.avg / 100), privacy_risks)

    if not os.path.exists(config_model_path):
        os.makedirs(config_model_path, exist_ok=True)

    checkpoint_path = os.path.join(config_model_path, "budget_{}_checkpoint.pth".format(int(budget)))

    new_checkpoint = {
        'epoch': int(budget),
        "1 - main_task_accuracy": 1 - (epoch_val_acc.avg / 100),
        'backbone_state_dict': BACKBONE.state_dict(),
        'head_state_dict': HEAD.state_dict(),
        'head': HEAD_NAME,
        'loss': LOSS_NAME,
        'optimizer': OPTIMIZER_NAME,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'lr': LR,
    }

    costs = {
        "1 - main_task_accuracy": 1 - (epoch_val_acc.avg / 100),
    }

    for i, task in enumerate(SENSITIVE_TASKS_NAMES):
        new_checkpoint['privacy_risk_{}'.format(task)] = privacy_risks[i]
        costs['privacy_risk_{}'.format(task)] = privacy_risks[i]

    torch.save(new_checkpoint, checkpoint_path)

    new_checkpoint.pop('backbone_state_dict', None)
    new_checkpoint.pop('head_state_dict', None)

    with open(os.path.join(config_model_path, 'costs_dict_{}.csv'.format(budget)), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in new_checkpoint.items():
            writer.writerow([key, value])

    print("COSTS:", costs)

    return costs


if __name__ == '__main__':
    args = parse_args()

    wandb.init(project="MultiObjective Optimization-Accuracy and Privacy Risk")

    cs = ConfigurationSpace()

    # HyperParameters to optimize
    lr_adam = Float("lr_adam", (1e-5, 1e-2), default=1e-3, log=True)
    lr_sgd = Float("lr_sgd", (0.0001, 1), log=True, default=0.1)
    if configurations[args.config_file]['HEAD_NAME'] == 'Baseline':
        head = Categorical("head", ["Baseline", "ArcFace", "CosFace"], default="Baseline")
    else:
        head = Categorical("head", ["CosFace", "ArcFace", "Linear"], default="CosFace")
    weight_decay = Float("weight_decay", (1e-6, 0.1), default=0.01, log=True)
    loss = Categorical("loss", ["Focal", "CrossEntropy"], default="Focal")
    optimizer = Categorical("optimizer", ["Adam", "SGD"], default="SGD")
    batch_size = Integer("batch_size", (30, 100), default=80)

    # Conditions
    use_lr_adam = InCondition(lr_adam, parent=optimizer, values=['Adam'])
    use_lr_sgd = InCondition(lr_sgd, parent=optimizer, values=['SGD'])

    cs.add_hyperparameters([
        lr_adam,
        lr_sgd,
        head,
        loss,
        optimizer,
        batch_size,
        weight_decay
    ])
    cs.add_conditions([use_lr_adam, use_lr_sgd])

    cfg_file = configurations[args.config_file]
    sensitive_tasks = cfg_file['SENSITIVE_TASK']
    objectives = ["1 - main_task_accuracy"]

    for task in sensitive_tasks:
        objectives.append("privacy_risk_{}".format(task))

    print("Objectives", objectives)

    # Scenario used for multi-objective optimization
    scenario = Scenario(
        cs,
        walltime_limit=400000000000,  # After x seconds, we stop the hyperparameter optimization
        n_trials=args.n_trials,  # Evaluate max 200 different trials
        objectives=objectives,
        min_budget=args.min_budget,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
        max_budget=args.max_budget,  # Train the MLP using a hyperparameter configuration for at most 120 epochs
        n_workers=3
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=args.init_configs)
    intensifier = MultiFidelityFacade.get_intensifier(scenario, eta=args.eta, incumbent_selection="highest_budget")
    multi_objective_algorithm = ParEGO(scenario)

    RESULTS_ROOT = cfg_file['RESULTS_ROOT']
    # model_path = os.path.join(args.exp_path + '-' + datetime.today().strftime('%Y-%m-%d'), "model_checkpoints")
    # config_path = os.path.join(args.exp_path + '-' + datetime.today().strftime('%Y-%m-%d'), "best_configs")
    # adv_config_path = os.path.join(args.exp_path + '-' + datetime.today().strftime('%Y-%m-%d'), "adv_config")
    model_path = os.path.join(args.exp_path, "model_checkpoints")
    config_path = os.path.join(args.exp_path, "best_configs")
    adv_config_path = os.path.join(args.exp_path, "adv_config")

    cost_function = functools.partial(train_bio_task, model_path=os.path.join(RESULTS_ROOT, model_path),
                                      adv_config_path=os.path.join(RESULTS_ROOT, adv_config_path),
                                      checkpointing=args.checkpointing, config_file=args.config_file)

    smac = MultiFidelityFacade(
        scenario,
        cost_function,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
        callbacks=[CustomCallback()],
        intensifier=intensifier
    )

    # train one model to generate adv config
    random_config = cs.sample_configuration()
    _ = train_bio_task(random_config, 0, int(args.max_budget / 2), model_path=os.path.join(RESULTS_ROOT, model_path),
                       adv_config_path=os.path.join(RESULTS_ROOT, adv_config_path),
                       checkpointing=args.checkpointing, config_file=args.config_file,
                       adv_search_mode=True)

    incumbent = smac.optimize()

    incumbent_df = pd.DataFrame(columns=[param.name for param in cs.get_hyperparameters()] + objectives + ['hash'])
    for inc in incumbent:
        cost = smac.runhistory.average_cost(inc)
        print("Incumbent", dict(inc))
        print("Cost", cost)
        values = dict(inc)
        for i, obj in enumerate(objectives):
            values[obj] = cost[i]
        values['hash'] = get_config_hash(inc)
        incumbent_df = incumbent_df.append(values, ignore_index=True)

    print(incumbent_df)

    directory = os.path.join(RESULTS_ROOT, config_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, "Time_{}_configs.csv".format(get_time()))

    incumbent_df.to_csv(filepath, index=False)
    if len(objectives) < 3:
        plot_pareto(smac, incumbent)


    print("Validated costs from the Pareto front (incumbents):")
    for inc in incumbent:
        print(inc)
        # cost = smac.validate(inc)
        cost = smac.runhistory.average_cost(inc)
        print("---", cost)
