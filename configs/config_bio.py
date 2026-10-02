"""Experiment configs for training soft-biometric (bio) main-task models."""

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
    "ArchTrans": {
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=97,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.1,  # initial LR
            NUM_EPOCH=120,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.002456,
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.000264,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.001741,
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
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RACE',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.001643,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.001751,
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
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RACE',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.001581,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.000108,
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.000754,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.000806,
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Baseline',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=4608,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.001307,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=0.000087,
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

    },

    "Exp2": {
        # Embed size variation
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=10,  # feature dimension
            BATCH_SIZE=16,
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=16,  # feature dimension
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=32,  # feature dimension
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=64,  # feature dimension
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=128,  # feature dimension
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
        6: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=256,  # feature dimension
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

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
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

    },

    # Head Experiment
    "Exp3": {
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='ArcFace',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.01,  # initial LR
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='CosFace',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.01,  # initial LR
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

    },

    # Head Experiment
    "Exp4": {
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.05,  # initial LR
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.01,  # initial LR
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.005,  # initial LR
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

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.001,  # initial LR
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

        6: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.5,  # initial LR
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

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.9,  # initial LR
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

    },

    # TRAINING FOUND INCUMBENTS
    "Incumbents": {
        # acc 0.114 priv 0.195
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Focal',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=70,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.896467,  # initial LR
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
        # ACC 0.050000000000000044 PRIV 0.27698108224933665
        2: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=80,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.002753672312748228,  # initial LR
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

        # ACC 0.07407409667968745 PRIV 0.20239137875464802
        3: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=80,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.90817501272949016,  # initial LR
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

        # ACC 0.080 PRIV 0.183
        4: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=100,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.000662,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
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

        #  ACC 0.050 PRIV 0.199
        5: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='GENDER',

            DATA_ROOT=f'{_REPO_ROOT}/data/FairFace/fairface-img-margin025-trainval',
            CSV_TRAIN_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_train.csv',
            CSV_VAL_PATH=f'{_REPO_ROOT}/data/FairFace/fairface_label_val.csv',
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models_bio',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='Softmax',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
            LOSS_NAME='Softmax',  # support: ['Focal', 'Softmax']
            MARGIN=None,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=40,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.184975,  # initial LR
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

    },
}
