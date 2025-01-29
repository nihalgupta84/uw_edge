#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List
import os
from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.WANDB = CN()
        self._C.WANDB.NAME = 'evup_01'
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.SESSION = 'UW'
        self._C.MODEL.INPUT = 'raw'
        self._C.MODEL.TARGET = 'ref'

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
        self._C.TRAINING.SAVE_DIR = 'checkpoints'  # Ensure this path is valid
        self._C.TRAINING.PS_W = 512
        self._C.TRAINING.PS_H = 512
        self._C.TRAINING.ORI = False
        self._C.TRAINING.LOG_FILE = 'log.txt'

        self._C.TESTING = CN()
        self._C.TESTING.INPUT = 'raw'  # Update to 'raw'
        self._C.TESTING.TARGET = 'ref'  # Update to 'ref'
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
        
        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        # self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
    
    def finalize_config(self):
        if self._C.WANDB.NAME:
            # Append WANDB.NAME to SAVE_DIR
            self._C.TRAINING.SAVE_DIR = os.path.join(self._C.TRAINING.SAVE_DIR, self._C.WANDB.NAME)
            # Use WANDB.NAME for LOG_FILE
            self._C.TRAINING.LOG_FILE = f"log_{self._C.WANDB.NAME}.txt"
            self._C.LOG.LOG_DIR = os.path.join(self._C.LOG.LOG_DIR, self._C.WANDB.NAME)
