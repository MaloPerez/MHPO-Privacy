"""Experiment configs for rerunning specific main/private-task combinations."""

import os

import torch

# Root directory data/log/model paths below are resolved against. Defaults to the
# repo root (so a fresh clone works out of the box); override with the
# FACEEVOLVE_DATA_ROOT environment variable to point at a different data location.
_REPO_ROOT = os.environ.get(
    'FACEEVOLVE_DATA_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
)

configurations = {

    1: dict(
        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=98,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.001,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0005835344923,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    2: dict(
        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=97,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.000263526404,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0017412429067,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

 3: dict(

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=99,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0016428187029,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.00175141142,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    4: dict(

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='CrossEntropy',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=94,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0015811221334,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0001082917249,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    5: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0013069615327,  # initial LR
        NUM_EPOCH=94,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=8.72720058e-05,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    6: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='CrossEntropy',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=98,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.000754294654,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0008064332822,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),


    #### SINGLE HPO ########################################################

    7: dict(
        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='CrossEntropy',  # support: ['Focal', 'Softmax']
        MARGIN=None,
        ADV_NAME='Baseline',
        ADV_LR=0.9,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=62,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0190891715849,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=6.2739276e-05,
        OPTIM='SGD',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    8: dict(
        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='CrossEntropy',  # support: ['Focal', 'Softmax']
        MARGIN=None,
        ADV_NAME='Baseline',
        ADV_LR=0.9,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=62,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0190891715849,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=6.2739276e-05,
        OPTIM='SGD',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    9: dict(
        MAIN_TASK='RACE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        MARGIN=None,
        ADV_NAME='Baseline',
        ADV_LR=0.9,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=95,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=3.70592664e-05,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0037424722173,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    10: dict(
        MAIN_TASK='RACE',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        MARGIN=None,
        ADV_NAME='Baseline',
        ADV_LR=0.9,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=95,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=3.70592664e-05,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=0.0037424722173,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),
    11: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.0013069615327,  # initial LR
        NUM_EPOCH=94,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=8.72720058e-05,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

    12: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/rerun_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=4608,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.00053069615327,  # initial LR
        NUM_EPOCH=94,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=8.72720058e-05,
        OPTIM='Adam',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=False,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0],  # specify your GPU ids
        PIN_MEMORY=True,
        NUM_WORKERS=50,
    ),

}
