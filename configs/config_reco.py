"""Experiment configs for face-recognition (identity) task training."""

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
    # Head Variation
    "Exp1": {
        # ArcFace
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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

        # CosFace
        2: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
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

        # Softmax
        3: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
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

        # MagFace
        4: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME='MagFace',  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
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
            WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
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

        # FaceNet
        5: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
            LOG_ROOT=f'{_REPO_ROOT}/logs',  # the root to log your train/val status
            BACKBONE_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint
            HEAD_RESUME_ROOT='./',  # the root to resume training from a saved checkpoint

            BACKBONE_NAME='IR_SE_50',
            # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101',
            # 'IR_SE_152']
            HEAD_NAME=None,  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'MagFace']
            LOSS_NAME='TripletLoss',  # support: ['Focal', 'Softmax', 'TripletLoss]
            MARGIN=0.3,

            INPUT_SIZE=[112, 112],  # support: [112, 112] and [224, 224]
            RGB_MEAN=[0.5, 0.5, 0.5],  # for normalize inputs to [-1, 1]
            RGB_STD=[0.5, 0.5, 0.5],
            EMBEDDING_SIZE=512,  # feature dimension
            BATCH_SIZE=80,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.001,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

    # LR Variation
    "Exp2": {

        # LR = 0.1
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # LR = 0.05
        2: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.05,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # LR = 0.01
        3: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # LR = 0.005
        4: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.005,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # LR = 0.001
        5: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.001,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
            MOMENTUM=0.9,
            STAGES=[35, 65, 95],  # epoch stages to decay learning rate

            DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            MULTI_GPU=False,
            # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
            # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
            GPU_ID=[0],  # specify your GPU ids
            PIN_MEMORY=True,
            NUM_WORKERS=50,
        )

    },

    # Embedding Size Variation
    "Exp3": {

        # Embedding Size = 64
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            EMBEDDING_SIZE=64,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # Embedding Size = 128
        2: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            EMBEDDING_SIZE=128,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # Embedding Size = 256
        3: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            EMBEDDING_SIZE=256,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # Embedding Size = 512
        4: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
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

        # Embedding Size = 1024
        5: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            EMBEDDING_SIZE=1024,  # feature dimension
            BATCH_SIZE=256,
            DROP_LAST=True,  # whether drop the last batch to ensure consistent batch_norm statistics
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,
            OPTIM='SGD',  # do not apply to batch_norm parameters
            MOMENTUM=0.9,
            STAGES=[35, 65, 95],  # epoch stages to decay learning rate

            DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            MULTI_GPU=False,
            # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export
            # CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
            GPU_ID=[0],  # specify your GPU ids
            PIN_MEMORY=True,
            NUM_WORKERS=50,
        )

    },

    # Optim Variation
    "Exp4": {

        # Optim = SGD
        1: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.1,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
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

        # Optim = Adam
        2: dict(
            SEED=1337,  # random seed for reproduce results

            MAIN_TASK='RECOGNITION',

            DATA_ROOT=f'{_REPO_ROOT}/data',
            DATASET_NAME='CASIA_WEB_FACE',
            # the parent root where your train/val/test data are stored
            MODEL_ROOT=f'{_REPO_ROOT}/trained_models',  # the root to buffer your checkpoints
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
            LR=0.005,  # initial LR
            NUM_EPOCH=125,  # total epoch number (use the firt 1/25 epochs to warm up)
            WEIGHT_DECAY=5e-4,  # do not apply to batch_norm parameters
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
}
