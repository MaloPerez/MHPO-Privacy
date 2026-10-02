"""Root-level training entry point for the face-recognition (identity) task."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from data_processing.data_loaders import TripletFaceDataset
from loss.triplet_loss import TripletLoss
from configs.config_reco import configurations
import argparse
from torch.nn.modules.distance import PairwiseDistance
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Softmax, MagFace
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

import wandb
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Trains Embed Networks')
    parser.add_argument('--config-file', type=int, default=0,
                        help='config file to use for the training net embedding')
    parser.add_argument('--exp-id', type=str, default="EXP1",
                        help='Experiment ID to choose config file [Exp1, Exp2, etc]')
    parser.add_argument('--model-path', type=str, default='trained_embed_net',
                        help='path at which to save the trained model (default: trained_model)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    wandb.init(project="EmbeddingNet Framework Training")

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[args.exp_id][args.config_file]

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    MAIN_TASK = cfg['MAIN_TASK']

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    DATASET_NAME = cfg['DATASET_NAME']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101',
    # 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax', 'TripletLoss']
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

    NUM_TRIPLETS = 400000

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    WAND_B_NAME = 'FaceNet' if (LOSS_NAME == 'TripletLoss') else HEAD_NAME

    writer = SummaryWriter(os.path.join(LOG_ROOT, args.model_path))  # writer for buffering intermedium results

    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in
        # online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN,
                             std=RGB_STD),
    ])

    if LOSS_NAME == 'TripletLoss':
        dataset_train = TripletFaceDataset(os.path.join(DATA_ROOT, DATASET_NAME),
                                           f'{_ROOT}/casia_df.csv', NUM_TRIPLETS,
                                           train_transform)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, DATASET_NAME), train_transform)

    # create a weighted random sampler to process imbalanced data
    if LOSS_NAME == 'TripletLoss':
        sampler = None
    else:
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=True
    )

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, \
    calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(DATA_ROOT)

    # ======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_101': ResNet_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'ResNet_152': ResNet_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_50': IR_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_101': IR_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_152': IR_152(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_101': IR_SE_101(INPUT_SIZE, EMBEDDING_SIZE),
                     'IR_SE_152': IR_SE_152(INPUT_SIZE, EMBEDDING_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    if LOSS_NAME != 'TripletLoss':
        NUM_CLASS = len(train_loader.dataset.classes)
        print("Number of Training Classes: {}".format(NUM_CLASS))

        HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'Softmax': Softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'MagFace': MagFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS)}
        HEAD = HEAD_DICT[HEAD_NAME]
        print("=" * 60)
        print(HEAD)
        print("{} Head Generated".format(HEAD_NAME))
        print("=" * 60)
    else:
        HEAD = None

    LOSS_DICT = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss(),
                 'TripletLoss': TripletLoss(MARGIN)}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

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

    if LOSS_NAME == 'TripletLoss':
        if OPTIMIZER_NAME == 'SGD':
            OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                   {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
        elif OPTIMIZER_NAME == "Adam":
            OPTIMIZER = optim.Adam([{'params': backbone_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                   {'params': backbone_paras_only_bn}], lr=LR)
    else:
        if OPTIMIZER_NAME == 'SGD':
            OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                  {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
        elif OPTIMIZER_NAME == "Adam":
            OPTIMIZER = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                                   {'params': backbone_paras_only_bn}], lr=LR)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

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

    # ======= train & validation & save checkpoint =======#
    DISP_FREQ = len(train_loader) // 100  # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    for epoch in range(NUM_EPOCH):  # start training process

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
        top5 = AverageMeter()

        for batch_sample in tqdm(iter(train_loader)):

            batch_size = 0

            if LOSS_NAME == 'TripletLoss':
                anc_img = batch_sample['anc_img'].to(DEVICE)
                pos_img = batch_sample['pos_img'].to(DEVICE)
                neg_img = batch_sample['neg_img'].to(DEVICE)

                batch_size = anc_img.size(0)

                anc_embed, pos_embed, neg_embed = BACKBONE(anc_img), BACKBONE(pos_img), \
                                                  BACKBONE(neg_img)
            else:
                inputs, labels = batch_sample
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).long()

                batch_size = inputs.size(0)

                features = BACKBONE(inputs)

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            # compute output
            if LOSS_NAME == 'TripletLoss':
                l2_dist = PairwiseDistance(2)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < MARGIN).cpu().numpy().flatten()
                hard_triplets = np.where(all == 1)
                if len(hard_triplets[0]) == 0:
                    continue

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                loss = LOSS(anc_hard_embed, pos_hard_embed, neg_hard_embed)

            else:
                if isinstance(HEAD, Softmax):
                    outputs = HEAD(features)
                else:
                    outputs = HEAD(features, labels)
                loss = LOSS(outputs, labels)

            # measure accuracy and record loss
            if LOSS_NAME != 'TripletLoss':
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                top1.update(prec1.data.item(), batch_size)
                top5.update(prec5.data.item(), batch_size)

            losses.update(loss.data.item(), batch_size)

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                if LOSS_NAME == 'TripletLoss':
                    print("=" * 60)
                    print('Epoch {}/{} Batch {}/{}\t'
                          'Training Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss=losses))
                    print("=" * 60)

                    wandb.log({'Batch Training Loss Val {}'.format(WAND_B_NAME): losses.val,
                               'Batch Training Loss Avg {}'.format(WAND_B_NAME): losses.avg}
                              )
                else:
                    print("=" * 60)
                    print('Epoch {}/{} Batch {}/{}\t'
                          'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss=losses, top1=top1,
                        top5=top5))
                    print("=" * 60)

                    wandb.log({'Batch Training Loss Val {}'.format(WAND_B_NAME): losses.val,
                               'Batch Training Loss Avg {}'.format(WAND_B_NAME): losses.avg,
                               'Batch Training Prec@1 Val {} '.format(WAND_B_NAME): top1.val,
                               'Batch Training Prec@1 Avg {} '.format(WAND_B_NAME): top1.avg,
                               'Batch Training Prec@5 Val {} '.format(WAND_B_NAME): top5.val,
                               'Batch Training Prec@5 Avg {} '.format(WAND_B_NAME): top5.avg}
                              )

            batch += 1  # batch index

        # training statistics per epoch (buffer for visualization)
        if LOSS_NAME == 'TripletLoss':
            epoch_loss = losses.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            print("=" * 60)
            print('Epoch: {}/{}\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, NUM_EPOCH, loss=losses))
        else:
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            print("=" * 60)
            print('Epoch: {}/{}\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))

        print("=" * 60)

        # perform validation & save checkpoints per epoch
        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE,
                                                                      BACKBONE, lfw, lfw_issame)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                               BATCH_SIZE, BACKBONE, cfp_ff,
                                                                               cfp_ff_issame)
        buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                               BATCH_SIZE, BACKBONE, cfp_fp,
                                                                               cfp_fp_issame)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, agedb, agedb_issame)
        buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, calfw, calfw_issame)
        buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                            BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
        buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
        accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE,
                                                                                  BATCH_SIZE, BACKBONE, vgg2_fp,
                                                                                  vgg2_fp_issame)
        buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
        print(
            "Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {},\
            CPLFW Acc: {}, VGG2_FP Acc: {}".format(
                epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw,
                accuracy_cplfw, accuracy_vgg2_fp))
        print("=" * 60)

        wandb.log({'Training Loss Val {}'.format(WAND_B_NAME): losses.val,
                   'Training Loss Avg {}'.format(WAND_B_NAME): losses.avg,
                   'Training Prec@1 Val {} '.format(WAND_B_NAME): top1.val,
                   'Training Prec@1 Avg {} '.format(WAND_B_NAME): top1.avg,
                   'Training Prec@5 Val {} '.format(WAND_B_NAME): top5.val,
                   'Training Prec@5 Avg {} '.format(WAND_B_NAME): top5.avg,
                   'Val Acc LFW ': accuracy_lfw,
                   'Val Acc CFP_FF ': accuracy_cfp_ff,
                   'Val Acc CFP_FP ': accuracy_cfp_fp,
                   'Val Acc AgeDB ': accuracy_agedb,
                   'Val Acc CALFW ': accuracy_calfw,
                   'Val Acc CPLFW ': accuracy_cplfw,
                   'Val Acc VGG2_FP ': accuracy_vgg2_fp,
                   'Epoch': epoch + 1}
                  )

        # save checkpoints per epoch
        directory = os.path.join(MODEL_ROOT, args.model_path, )
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(directory,
                                                                  "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                                      BACKBONE_NAME, epoch + 1, batch, get_time())))
            if HEAD:
                torch.save(HEAD.state_dict(), os.path.join(directory,
                                                           "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                               HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(directory,
                                                           "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                               BACKBONE_NAME, epoch + 1, batch, get_time())))
            if HEAD:
                torch.save(HEAD.state_dict(), os.path.join(directory,
                                                           "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                                               HEAD_NAME, epoch + 1, batch, get_time())))
