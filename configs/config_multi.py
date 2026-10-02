"""Experiment configs for the multi-objective HPO (M-HPO/ParEGO) search, one entry
per main-task/private-task/architecture/dataset combination."""

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
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    2: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    3: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race', 'age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
    4: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
    5: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
        OPTIM='SGD',
        MOMENTUM=0.9,
        STAGES=[35, 65, 95],  # epoch stages to decay learning rate

        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True,
        # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
        # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID=[0, 1],  # specify your GPU ids
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
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
    7: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='Baseline',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='Baseline',
        ADV_LR=0.2,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
    10: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    11: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    12: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    ###### ResNet #######

    13: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    14: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='GENDER',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    15: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    16: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='RACE',
        SENSITIVE_TASK=['age'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    17: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['gender'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    18: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='AGE',
        SENSITIVE_TASK=['race'],

        DATASET='FairFace',
        DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    # CelebA

    19: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Male',
        SENSITIVE_TASK=['Eyeglasses'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    20: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Eyeglasses',
        SENSITIVE_TASK=['Male'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    21: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Male',
        SENSITIVE_TASK=['Wearing_Hat'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    22: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Wearing_Hat',
        SENSITIVE_TASK=['Male'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    23: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Male',
        SENSITIVE_TASK=['Gray_Hair'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    24: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Gray_Hair',
        SENSITIVE_TASK=['Male'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
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
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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

    # Single-hyperparameter validation sweep target: IR_SE_50 + CelebA, never paired in this project (all other
    # CelebA entries above use the Baseline backbone). Same Male<->Eyeglasses task
    # pairing as indices 19/20 (Baseline+CelebA) so this isolates the architecture
    # generalization question instead of conflating it with a new task pairing too.
    25: dict(
        SEED=1337,  # random seed for reproduce results

        MAIN_TASK='Male',
        SENSITIVE_TASK=['Eyeglasses'],

        DATASET='CelebA',
        DATA_ROOT=f'{_REPO_ROOT}/data',
        CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
        CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
        RESULTS_ROOT=f'{_REPO_ROOT}/multiHPO_results',  # the root to buffer your checkpoints
        LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status

        BACKBONE_NAME='IR_SE_50',
        # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
        # 'IR_SE_152']
        HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
        ADV_NAME='5Layer',
        ADV_LR=0.9,
        MARGIN=None,

        INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
        RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512,  # feature dimension
        BATCH_SIZE=256,
        DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
        LR=0.1,  # initial LR
        NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY=5e-4,
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
}
