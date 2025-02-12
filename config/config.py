#File: config/config.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management using yacs ConfigNode
"""
from typing import Any, List
import os
from yacs.config import CfgNode as CN

class Config(object):
    r"""
    A collection of all required config parameters, loaded from a YAML file and optionally overridden.

    Usage:
        config = Config("config.yml", ["MODEL.SESSION", "mySession"])
        config.finalize_config()
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        # Base container
        self._C = CN()

        # Minimal defaults
        self._C.WANDB = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.SESSION = 'UW'
        self._C.MODEL.INPUT = 'raw'
        self._C.MODEL.TARGET = 'ref'
        self._C.MODEL.INPUT_CHANNELS = 3
        self._C.MODEL.BASE_CHANNELS = 64
        self._C.MODEL.EDGE_MODULE = True
        self._C.MODEL.ATTENTION_MODULE = True
        self._C.MODEL.EDGE_VK = True
        self._C.MODEL.EDGE_HK = True
        self._C.MODEL.EDGE_CK = True
        self._C.MODEL.INIT_WEIGHTS = True
        # NEW: Specify default model type
        self._C.MODEL.NAME = 'version3'
        self._C.MODEL.DATASET_NAME = 'UIEB'
        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 4
        self._C.OPTIM.SEED = 3407
        self._C.OPTIM.NUM_EPOCHS = 300
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.WANDB = False

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.WEIGHT = None
        self._C.TRAINING.TRAIN_DIR = './image_dirs/EVUP/train'
        self._C.TRAINING.VAL_DIR = './image_dirs/EVUP/val'
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.PS_W = 512
        self._C.TRAINING.PS_H = 512
        self._C.TRAINING.ORI = False
        self._C.TRAINING.LOG_FILE = 'log.txt'
        # NEW: Add gradient clipping and mixed precision options.
        self._C.TRAINING.CLIP_GRAD = 1.0
        self._C.TRAINING.MIXED_PRECISION = "no"  # Options: "no", "fp16", "bf16"

        self._C.TESTING = CN()
        self._C.TESTING.INPUT = 'raw'
        self._C.TESTING.TARGET = 'ref'
        self._C.TESTING.VAL_DIR = 'image_dirs/EVUP/val'
        self._C.TESTING.WEIGHT = None
        self._C.TESTING.SAVE_IMAGES = True
        self._C.TESTING.RESULT_DIR = 'result'
        self._C.TESTING.LOG_FILE = 'log.txt'

        self._C.LOG = CN()
        self._C.LOG.LOG_DIR = 'output_dir'

        # Losses block
        self._C.LOSSES = CN()
        self._C.LOSSES.USE_PSNR = True
        self._C.LOSSES.PSNR_SCALE = 1.0
        self._C.LOSSES.USE_SSIM = True
        self._C.LOSSES.SSIM_SCALE = 0.3
        self._C.LOSSES.USE_LPIPS = True
        self._C.LOSSES.LPIPS_SCALE = 0.7
        self._C.LOSSES.USE_EDGE = True
        self._C.LOSSES.EDGE_SCALE = 0.1
        self._C.LOSSES.USE_FREQ = True
        self._C.LOSSES.FREQ_SCALE = 0.1

        # Merge from YAML first, then from override list
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

    def dump(self, file_path: str):
        """Saves config to YAML at `file_path`."""
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()

    def finalize_config(self):
        """
        Makes final adjustments. If self._C.WANDB.NAME is non-empty,
        it appends that to the training SAVE_DIR and modifies the LOG_FILE.
        """
        if self._C.MODEL.SESSION:
            self._C.TRAINING.SAVE_DIR = os.path.join(
                self._C.TRAINING.SAVE_DIR,
                self._C.MODEL.NAME,
                self._C.MODEL.DATASET_NAME,
                self._C.MODEL.SESSION
            )
            self._C.TRAINING.LOG_FILE = f"log_{self._C.MODEL.DATASET_NAME}_{self._C.MODEL.NAME}_{self._C.MODEL.SESSION}.txt"
            self._C.LOG.LOG_DIR = os.path.join(
                self._C.LOG.LOG_DIR,
                self._C.MODEL.NAME,
                self._C.MODEL.DATASET_NAME,
                self._C.MODEL.SESSION
            )
            self._C.TESTING.RESULT_DIR = os.path.join(
                self._C.TESTING.RESULT_DIR, self._C.MODEL.NAME, self._C.MODEL.DATASET_NAME, self._C.MODEL.SESSION)
            self.C.TESTING.LOG_DIR = os.path.join(
                self._C.TESTING.LOG_DIR, self._C.MODEL.NAME, self._C.MODEL.DATASET_NAME, self._C.MODEL.SESSION)
            self._C.TESTING.LOG_FILE = f"test_{self._C.MODEL.DATASET_NAME}_{self._C.MODEL.SESSION}_{self._C.MODEL.NAME}.txt"