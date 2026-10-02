"""Adversarial Representation Learning (ARL) privacy-defense baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _d in ['backbone', 'data_processing', 'util', 'head', 'loss', 'configs', 'MAC']:
    sys.path.insert(0, os.path.join(_ROOT, _d))

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
    parser = argparse.ArgumentParser(description='Trains Embed Networks for Bio Task')
    parser.add_argument('--config-file', type=int, default=0,
                        help='config file to use for the training net embedding')
    parser.add_argument('--exp-id', type=str, default="EXP1",
                        help='Experiment ID to choose config file [Exp1, Exp2, etc]')
    parser.add_argument('--model-path', type=str, default='trained_embed_net_bio',
                        help='path at which to save the trained model (default: trained_model)')
    args = parser.parse_args()
    return args


def validation_bio(val_loader, DEVICE, LOSS_NAME, BACKBONE, MAIN_TASK, HEAD, LOSS, label_index_dict, DATASET='FairFace'):
    batch = 0
    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_sample in tqdm(iter(val_loader)):

        if DATASET == 'FairFace':
            inputs = batch_sample['img'].to(DEVICE)
            labels = batch_sample[MAIN_TASK.lower()].reshape((-1)).to(DEVICE)
        elif DATASET == 'CelebA':
            label_index = label_index_dict[MAIN_TASK]
            inputs = batch_sample[0].to(DEVICE)
            labels = batch_sample[1][:, label_index].to(DEVICE)

        batch_size = inputs.size(0)

        features = BACKBONE(inputs)

        if isinstance(HEAD, Softmax) or isinstance(HEAD, Public_Classifier):
            outputs = HEAD(features)
        else:
            outputs = HEAD(features, labels)
        loss = LOSS(outputs, labels)

        # measure val accuracy and val loss
        if LOSS_NAME != 'TripletLoss':
            prec1 = accuracy(outputs.data, labels)
            top1.update(prec1[0].item(), batch_size)

        losses.update(loss.data.item(), batch_size)

        batch += 1  # batch index

    epoch_val_loss = losses
    epoch_val_acc = top1

    return epoch_val_loss, epoch_val_acc


def aux_adversary_privacy_risk(embedding_net, embedding_size, train_dataset, test_dataset, device, sensitive_task,
                               ADV_NAME, DATASET, label_index_dict = None, weights_train = None):
    adv_batch_size = 128
    adv_epochs = 10
    learning_rate = ADV_LR

    sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=adv_batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=adv_batch_size, sampler=sampler, pin_memory=True,
        num_workers=50, drop_last=False, shuffle=True
    )

    NUM_CLASS_DICT = {'gender': 2,
                      'race': 7,
                      'age': 9}
    CLASS_WEIGHT_IDX = {'gender': 0,
                        'race': 1,
                        'age': 2}

    NUM_CLASS = NUM_CLASS_DICT.get(sensitive_task, 2)

    priv_risks = []
    learning_rates = [1, 0.5, 0.1, 0.01]

    for lr in learning_rates:

        ADV_DICT = {
            '5Layer': BioTask_5Layers(in_features=embedding_size, out_features=NUM_CLASS).to(device),
            'Baseline': Public_Classifier(output=NUM_CLASS).to(device)
        }
        adv_model = ADV_DICT[ADV_NAME]

        # loss_crit_adv = FocalLoss()
        if DATASET == 'FairFace':
            loss_crit_adv = torch.nn.CrossEntropyLoss(weight=train_dataset.class_weights[CLASS_WEIGHT_IDX[sensitive_task]]
                                                  .to(device))
        elif DATASET == 'CelebA':
            loss_crit_adv = torch.nn.CrossEntropyLoss(weight=weights_train[label_index_dict[sensitive_task]]
                                                      .to(device))

        optim_adv = optim.SGD(adv_model.parameters(), lr=lr)

        for e in range(adv_epochs):

            for idx, batch in enumerate(train_loader):
                adv_model.train()
                optim_adv.zero_grad()

                if DATASET == 'FairFace':
                    imgs = batch['img'].to(DEVICE)
                    target = batch[sensitive_task].reshape((-1)).to(DEVICE)
                elif DATASET == 'CelebA':
                    label_index = label_index_dict[sensitive_task]
                    imgs = batch[0].to(DEVICE)
                    target = batch[1][:,label_index].to(DEVICE)

                with torch.no_grad():
                    input_adv = embedding_net(imgs).to(device)

                forward_adv = adv_model.forward(input_adv).to(device)
                loss_adv = loss_crit_adv.forward(forward_adv, target)

                loss_adv.backward()

                optim_adv.step()

        all_preds = []
        all_targets = []

        for idx, batch in enumerate(test_loader):
            adv_model.eval()

            if DATASET == 'FairFace':
                imgs = batch['img'].to(DEVICE)
                target = batch[sensitive_task].reshape((-1)).to(DEVICE)
            elif DATASET == 'CelebA':
                label_index = label_index_dict[sensitive_task]
                imgs = batch[0].to(DEVICE)
                target = batch[1][:, label_index].to(DEVICE)

            with torch.no_grad():
                input_adv = embedding_net(imgs).to(device)

            forward_adv = adv_model.forward(input_adv).to(device)

            all_preds = all_preds + forward_adv.topk(1)[1].cpu().tolist()
            all_targets = all_targets + target.tolist()

        bal_acc_adv = balanced_accuracy_score(all_targets, all_preds)
        # PR = adversary_balanced_accuracy - chance_level (eq. 2 in paper)
        chance_level = 1.0 / NUM_CLASS
        priv_risks.append(bal_acc_adv - chance_level)

    print(priv_risks)
    return max(priv_risks)


if __name__ == '__main__':
    args = parse_args()

    wandb.init(project="Adversarial Learning")

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[args.exp_id][args.config_file]

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    DATASET = cfg['DATASET']
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    CSV_TRAIN_PATH = cfg['CSV_TRAIN_PATH']
    CSV_VAL_PATH = cfg['CSV_VAL_PATH']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101',
    # 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax', 'TripletLoss']
    ADV_NAME = cfg['ADV_NAME']
    MARGIN = cfg['MARGIN']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    OPTIMIZER_NAME = cfg['OPTIM']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    ADV_LR = cfg['ADV_LR']  # 0.9
    LAMBDA = cfg['LAMBDA']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    MAIN_TASK = cfg['MAIN_TASK']

    NUM_CLASS_DICT = {'age': 9,
                      'race': 7,
                      'gender': 2}

    if DATASET == 'CelebA':
        NUM_CLASS = 2
    else:
        NUM_CLASS = NUM_CLASS_DICT[MAIN_TASK.lower()]

    SENSITIVE_TASK = cfg['SENSITIVE_TASK']

    NUM_SENSITIVE_CLASS = NUM_CLASS_DICT.get(SENSITIVE_TASK, 2)

    CLASS_WEIGHT_IDX = {'gender': 0,
                        'race': 1,
                        'age': 2}

    writer = SummaryWriter(os.path.join(LOG_ROOT, args.model_path))  # writer for buffering intermedium results

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
    INDEX_LABEL = None

    if DATASET == 'CelebA':
        dataset_train, attr, weights_train = load_celebA_data(DATA_ROOT, "train", "attr", train_transform)
        dataset_val, _, weights_val = load_celebA_data(DATA_ROOT, "valid", "attr", train_transform)

        INDEX_LABEL = dict(zip(attr, range(len(attr))))
        MAIN_TASK = INDEX_LABEL[MAIN_TASK]
        WEIGHTS = weights_train[MAIN_TASK].to(DEVICE)
        SENSITIVE_TASK = INDEX_LABEL[SENSITIVE_TASK]
        print("SENSITIVE TASKS NAMES 1:", SENSITIVE_TASK)
    else:
        dataset_train = FairFaceDataset(DATA_ROOT, CSV_TRAIN_PATH, train_transform)
        dataset_val = FairFaceDataset(DATA_ROOT, CSV_VAL_PATH, train_transform)

    pre_train_size = int(0.5 * len(dataset_train))
    adv_train_size = len(dataset_train) - pre_train_size
    pre_train_dataset, adv_train_dataset = torch.utils.data.random_split(dataset_train,
                                                                         [pre_train_size, adv_train_size])

    sampler = None

    pre_train_loader = torch.utils.data.DataLoader(
        pre_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )

    adv_train_loader = torch.utils.data.DataLoader(
        adv_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(BATCH_SIZE / 2), sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )

    # ======= model & loss & optimizer =======#
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
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Softmax': Softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                 'Baseline': Public_Classifier(output=NUM_CLASS).to(DEVICE)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    # adversarial classifier
    ADV_DICT = {
        '5Layer': BioTask_5Layers(in_features=EMBEDDING_SIZE, out_features=NUM_SENSITIVE_CLASS).to(DEVICE),
        'Baseline': Public_Classifier(output=NUM_SENSITIVE_CLASS).to(DEVICE)
    }
    adv_classifier = ADV_DICT[ADV_NAME]

    LOSS_DICT = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    # adversarial loss
    if DATASET == 'FairFace':
        ADV_LOSS = torch.nn.CrossEntropyLoss(weight=dataset_train.class_weights[CLASS_WEIGHT_IDX[SENSITIVE_TASK]]
                                         .to(DEVICE))
    elif DATASET == 'CelebA':
        ADV_LOSS = torch.nn.CrossEntropyLoss(weight=weights_train[SENSITIVE_TASK]
                                         .to(DEVICE))

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

    if OPTIMIZER_NAME == 'SGD':
        OPTIMIZER_ENCODER = optim.SGD(
            [{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
             {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
        OPTIMIZER_HEAD = optim.SGD([{'params': head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}], lr=LR,
                                   momentum=MOMENTUM)
    elif OPTIMIZER_NAME == "Adam":
        OPTIMIZER_ENCODER = optim.Adam([{'params': backbone_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                        {'params': backbone_paras_only_bn}], lr=LR)
        OPTIMIZER_HEAD = optim.Adam([{'params': head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}], lr=LR)

    print("=" * 60)
    print(OPTIMIZER_ENCODER)
    print(OPTIMIZER_HEAD)
    print("Optimizer Generated")
    print("=" * 60)

    # adversarial optimizer
    ADV_OPTIMIZER = optim.SGD(adv_classifier.parameters(), lr=ADV_LR)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(
                BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    # ======================================================#
    # ===================PRE-TRAINING=======================#
    # ======================================================#

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = len(pre_train_loader) // 100  # frequency to display training loss & acc

    NUM_PRE_TRAIN_EPOCH = 60

    NUM_EPOCH_WARM_UP = NUM_PRE_TRAIN_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(pre_train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up

    for epoch in range(NUM_PRE_TRAIN_EPOCH):
        if epoch == STAGES[0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR
            # manually (with slight modification) once plateau observed
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)

        BACKBONE.train()  # set to training mode
        if HEAD:
            HEAD.train()

        losses_main = AverageMeter()

        top1_main = AverageMeter()

        batch = 0

        for batch_sample in tqdm(iter(pre_train_loader)):

            if DATASET == 'CelebA':
                inputs = batch_sample[0].to(DEVICE)
                labels = batch_sample[1][:, MAIN_TASK].reshape((-1)).to(DEVICE)
            else:
                inputs = batch_sample['img'].to(DEVICE)
                labels = batch_sample[MAIN_TASK.lower()].reshape((-1)).to(DEVICE)

            batch_size = inputs.size(0)

            OPTIMIZER_ENCODER.zero_grad()
            OPTIMIZER_HEAD.zero_grad()

            features = BACKBONE(inputs)

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER_ENCODER)
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER_HEAD)

                # compute output

            if isinstance(HEAD, Softmax) or isinstance(HEAD, Public_Classifier):
                outputs = HEAD(features)
            else:
                outputs = HEAD(features, labels)

            loss_main = LOSS(outputs, labels)

            # measure accuracy and record loss
            acc_main = accuracy(outputs.data, labels)
            top1_main.update(acc_main[0].item(), batch_size)

            losses_main.update(loss_main.data.item(), batch_size)

            # compute gradient and do SGD step
            loss_main.backward()
            OPTIMIZER_ENCODER.step()
            OPTIMIZER_HEAD.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss_main.val:.4f} ({loss_main.avg:.4f})\t'
                      'Training Prec@1 {top1_main.val:.3f} ({top1_main.avg:.3f})'.format(
                    epoch + 1, NUM_PRE_TRAIN_EPOCH, batch + 1, len(pre_train_loader) * NUM_PRE_TRAIN_EPOCH,
                    loss_main=losses_main,
                    top1_main=top1_main))
                print("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)

        epoch_loss = losses_main.avg
        epoch_acc = top1_main.avg
        epoch_val_loss, epoch_val_acc = validation_bio(val_loader, DEVICE, LOSS_NAME, BACKBONE, MAIN_TASK, HEAD, LOSS, INDEX_LABEL)
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Validation_Loss", epoch_val_loss.avg, epoch + 1)
        writer.add_scalar("Validation_Accuracy", epoch_val_acc.avg, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss_main.val:.4f} ({loss_main.avg:.4f})\t'
              'Training Prec@1 {top1_main.val:.3f} ({top1_main.avg:.3f})\t'
              'Validation Loss {valloss.val:.4f} ({valloss.avg:.3f})\t'
              'Validation Prec@1 {valtop1.val:.3f} ({valtop1.avg:.3f})'.format(epoch + 1, NUM_PRE_TRAIN_EPOCH,
                                                                               loss_main=losses_main,
                                                                               top1_main=top1_main,
                                                                               valloss=epoch_val_loss,
                                                                               valtop1=epoch_val_acc))

        print("=" * 60)

        wandb.log({#'Training Loss Val {}'.format(args.exp_id): losses_main.val,
                   'Training Loss Avg {}'.format(args.exp_id): losses_main.avg,
                   #'Training Prec@1 Val {} '.format(args.exp_id): top1_main.val,
                   'Training Prec@1 Avg {} '.format(args.exp_id): top1_main.avg,
                   #'Validation Loss Val {}'.format(args.exp_id): epoch_val_loss.val,
                   'Validation Loss Avg {}'.format(args.exp_id): epoch_val_loss.avg,
                   #'Validation Prec@1 Val {} '.format(args.exp_id): epoch_val_acc.val,
                   'Validation Prec@1 Avg {} '.format(args.exp_id): epoch_val_acc.avg,
                   'Epoch': epoch + 1}
                  )

        print("=" * 60)
        # Validation

        # save checkpoints per epoch
        directory = os.path.join(MODEL_ROOT, args.model_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # ======================================================#
    # ===================ADV-PRE-TRAINING===================#
    # ======================================================#

    NUM_ADV_PRE_TRAIN_EPOCH = 40

    for epoch in range(NUM_ADV_PRE_TRAIN_EPOCH):  # start training process

        adv_classifier.train()

        losses_adv = AverageMeter()

        top1_adv = AverageMeter()

        batch = 0

        for batch_sample in tqdm(iter(pre_train_loader)):

            if DATASET == 'CelebA':
                inputs = batch_sample[0].to(DEVICE)
                sensitive_labels = batch_sample[1][:, SENSITIVE_TASK].reshape((-1)).to(DEVICE)
            else:
                inputs = batch_sample['img'].to(DEVICE)
                sensitive_labels = batch_sample[SENSITIVE_TASK].reshape((-1)).to(DEVICE)

            batch_size = inputs.size(0)

            ADV_OPTIMIZER.zero_grad()

            features = BACKBONE(inputs)

            adv_output = adv_classifier.forward(features)

            loss_adv = ADV_LOSS.forward(adv_output, sensitive_labels)

            acc_adv = accuracy(adv_output.data, sensitive_labels)
            top1_adv.update(acc_adv[0].item(), batch_size)

            losses_adv.update(loss_adv.data.item(), batch_size)

            loss_adv.backward()
            ADV_OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Adversarial Loss {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
                      'Adversarial Prec@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                    epoch + 1, NUM_ADV_PRE_TRAIN_EPOCH, batch + 1, len(pre_train_loader) * NUM_ADV_PRE_TRAIN_EPOCH,
                    loss_adv=losses_adv,
                    top1_adv=top1_adv))
                print("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)

        epoch_adv_loss = losses_adv.avg
        epoch_adv_acc = top1_adv.avg
        writer.add_scalar("Training_Adversarial_Loss", epoch_adv_loss, epoch + 1)
        writer.add_scalar("Training_Adversarial_Accuracy", epoch_adv_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Adversarial Loss {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
              'Training Adversarial Prec@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})\t'.format(epoch + 1,
                                                                                             NUM_ADV_PRE_TRAIN_EPOCH,
                                                                                             loss_adv=losses_adv,
                                                                                             top1_adv=top1_adv))

        print("=" * 60)

        wandb.log({
            #'Training Adversarial Loss Val {}'.format(args.exp_id): losses_adv.val,
            'Training Adversarial Loss Avg {}'.format(args.exp_id): losses_adv.avg,
            #'Training Adversarial Prec@1 Val {} '.format(args.exp_id): top1_adv.val,
            'Training Adversarial Prec@1 Avg {} '.format(args.exp_id): top1_adv.avg,
            'Epoch': epoch + 1}
        )

        print("=" * 60)
        # Validation

        # save checkpoints per epoch
        directory = os.path.join(MODEL_ROOT, args.model_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # ======================================================#
    # ===================ADV-TRAINING=======================#
    # ======================================================#

    DISP_FREQ = len(adv_train_loader) // 100  # frequency to display training loss & acc

    TOTAL_EPOCHS = NUM_PRE_TRAIN_EPOCH + NUM_EPOCH

    NUM_EPOCH_WARM_UP = NUM_PRE_TRAIN_EPOCH + (NUM_EPOCH // 25)  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(adv_train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up

    # Resetting LRs
    schedule_lr(OPTIMIZER_ENCODER, LR)
    schedule_lr(OPTIMIZER_HEAD, LR)
    schedule_lr(ADV_OPTIMIZER, ADV_LR)

    for epoch in range(NUM_PRE_TRAIN_EPOCH, TOTAL_EPOCHS):  # start training process

        if epoch == STAGES[0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR
            # manually (with slight modification) once plateau observed
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)
            schedule_lr(ADV_OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)
            schedule_lr(ADV_OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER_ENCODER)
            schedule_lr(OPTIMIZER_HEAD)
            schedule_lr(ADV_OPTIMIZER)

        BACKBONE.train()  # set to training mode
        if HEAD:
            HEAD.train()
        adv_classifier.train()

        losses_main = AverageMeter()
        losses_adv = AverageMeter()

        top1_main = AverageMeter()
        top1_adv = AverageMeter()

        batch = 0

        for batch_sample in tqdm(iter(adv_train_loader)):

            if DATASET == 'CelebA':
                inputs = batch_sample[0].to(DEVICE)
                labels = batch_sample[1][:, MAIN_TASK].reshape((-1)).to(DEVICE)
                sensitive_labels = batch_sample[1][:, SENSITIVE_TASK].reshape((-1)).to(DEVICE)
            else:
                inputs = batch_sample['img'].to(DEVICE)
                labels = batch_sample[MAIN_TASK.lower()].reshape((-1)).to(DEVICE)
                sensitive_labels = batch_sample[SENSITIVE_TASK].reshape((-1)).to(DEVICE)


            batch_size = inputs.size(0)

            OPTIMIZER_ENCODER.zero_grad()
            OPTIMIZER_HEAD.zero_grad()
            ADV_OPTIMIZER.zero_grad()

            features = BACKBONE(inputs)

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER_ENCODER)
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER_HEAD)
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, ADV_OPTIMIZER)

                # compute output

            if isinstance(HEAD, Softmax) or isinstance(HEAD, Public_Classifier):
                outputs = HEAD(features)
            else:
                outputs = HEAD(features, labels)

            adv_output = adv_classifier.forward(features)

            loss_main = LOSS(outputs, labels)
            loss_adv = ADV_LOSS.forward(adv_output, sensitive_labels)

            # measure accuracy and record loss
            acc_main = accuracy(outputs.data, labels)
            top1_main.update(acc_main[0].item(), batch_size)

            acc_adv = accuracy(adv_output.data, sensitive_labels)
            top1_adv.update(acc_adv[0].item(), batch_size)

            losses_main.update(loss_main.data.item(), batch_size)
            losses_adv.update(loss_adv.data.item(), batch_size)

            par = LAMBDA
            # if epoch > NUM_EPOCH - par:
            #    par = NUM_EPOCH - epoch
            # print('par:', par)

            # compute gradient and do SGD step — alternating updates every 3 batches
            if batch % 3 == 0:
                E_loss = loss_main - par * loss_adv
                E_loss.backward()
                OPTIMIZER_ENCODER.step()
            elif batch % 3 == 1:
                loss_adv.backward()
                ADV_OPTIMIZER.step()
            else:
                loss_main.backward()
                OPTIMIZER_HEAD.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss_main.val:.4f} ({loss_main.avg:.4f})\t'
                      'Training Prec@1 {top1_main.val:.3f} ({top1_main.avg:.3f})\t'
                      'Adversarial Loss {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
                      'Adversarial Prec@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})'.format(
                    epoch + 1, TOTAL_EPOCHS, batch + 1, len(adv_train_loader) * TOTAL_EPOCHS, loss_main=losses_main,
                    top1_main=top1_main, loss_adv=losses_adv, top1_adv=top1_adv))
                print("=" * 60)

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)

        epoch_loss = losses_main.avg
        epoch_acc = top1_main.avg
        epoch_adv_loss = losses_adv.avg
        epoch_adv_acc = top1_adv.avg
        epoch_val_loss, epoch_val_acc = validation_bio(val_loader, DEVICE, LOSS_NAME, BACKBONE, MAIN_TASK, HEAD, LOSS, INDEX_LABEL)
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Training_Adversarial_Loss", epoch_adv_loss, epoch + 1)
        writer.add_scalar("Training_Adversarial_Accuracy", epoch_adv_acc, epoch + 1)
        writer.add_scalar("Validation_Loss", epoch_val_loss.avg, epoch + 1)
        writer.add_scalar("Validation_Accuracy", epoch_val_acc.avg, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss_main.val:.4f} ({loss_main.avg:.4f})\t'
              'Training Prec@1 {top1_main.val:.3f} ({top1_main.avg:.3f})\t'
              'Training Adversarial Loss {loss_adv.val:.4f} ({loss_adv.avg:.4f})\t'
              'Training Adversarial Prec@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})\t'
              'Validation Loss {valloss.val:.4f} ({valloss.avg:.3f})\t'
              'Validation Prec@1 {valtop1.val:.3f} ({valtop1.avg:.3f})'.format(epoch + 1, TOTAL_EPOCHS,
                                                                               loss_main=losses_main,
                                                                               loss_adv=losses_adv, top1_adv=top1_adv,
                                                                               top1_main=top1_main,
                                                                               valloss=epoch_val_loss,
                                                                               valtop1=epoch_val_acc))

        print("=" * 60)

        wandb.log({#'Training Loss Val {}'.format(args.exp_id): losses_main.val,
                   'Training Loss Avg {}'.format(args.exp_id): losses_main.avg,
                   #'Training Prec@1 Val {} '.format(args.exp_id): top1_main.val,
                   'Training Prec@1 Avg {} '.format(args.exp_id): top1_main.avg,
                   #'Training Adversarial Loss Val {}'.format(args.exp_id): losses_adv.val,
                   'Training Adversarial Loss Avg {}'.format(args.exp_id): losses_adv.avg,
                   #'Training Adversarial Prec@1 Val {} '.format(args.exp_id): top1_adv.val,
                   'Training Adversarial Prec@1 Avg {} '.format(args.exp_id): top1_adv.avg,
                   #'Validation Loss Val {}'.format(args.exp_id): epoch_val_loss.val,
                   'Validation Loss Avg {}'.format(args.exp_id): epoch_val_loss.avg,
                   #'Validation Prec@1 Val {} '.format(args.exp_id): epoch_val_acc.val,
                   'Validation Prec@1 Avg {} '.format(args.exp_id): epoch_val_acc.avg,
                   'Epoch': epoch + 1}
                  )

        print("=" * 60)
        # Validation

        # save checkpoints per epoch
        directory = os.path.join(MODEL_ROOT, args.model_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if epoch == (TOTAL_EPOCHS - 1):
            if MULTI_GPU:
                torch.save(BACKBONE.module.state_dict(), os.path.join(directory,
                                                                      "Backbone_{}_Epoch_{}_checkpoint.pth".format(
                                                                          BACKBONE_NAME, epoch + 1)))
                if HEAD:
                    torch.save(HEAD.state_dict(), os.path.join(directory,
                                                               "Head_{}_Epoch_{}_checkpoint.pth".format(
                                                                   HEAD_NAME, epoch + 1)))
                torch.save(adv_classifier.state_dict(), os.path.join(directory,
                                                                     "Adv_{}_Epoch_{}_checkpoint.pth".format(
                                                                         HEAD_NAME, epoch + 1)))
            else:
                torch.save(BACKBONE.state_dict(), os.path.join(directory,
                                                               "Backbone_{}_Epoch_{}_checkpoint.pth".format(
                                                                   BACKBONE_NAME, epoch + 1)))
                if HEAD:
                    torch.save(HEAD.state_dict(), os.path.join(directory,
                                                               "Head_{}_Epoch_{}_checkpoint.pth".format(
                                                                   HEAD_NAME, epoch + 1)))
                torch.save(adv_classifier.state_dict(), os.path.join(directory,
                                                                     "Adv_{}_Epoch_{}_checkpoint.pth".format(
                                                                         BACKBONE_NAME, epoch + 1)))

    final_privacy_risk = aux_adversary_privacy_risk(BACKBONE, EMBEDDING_SIZE, dataset_train, dataset_val,
                                                    DEVICE, SENSITIVE_TASK, ADV_NAME, DATASET,
                                                    label_index_dict=INDEX_LABEL, weights_train=WEIGHTS)

    wandb.log({'Final Privacy Risk {}'.format(args.exp_id): final_privacy_risk})