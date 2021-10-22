"""
Experiment related stuffs
Act as a bridge between main and utils (logging, init directory, etc)
"""
from pathlib import Path
from utils.config import parse_json
from utils.logger import init_logger

import os

import random
import numpy as np
import torch

def init_experiment(args):
    """
    in:
        args: arguments such as hyperparameters and other
    out:
        --
    procedure to initialize experiment consisting of:
        - parse config file as a json dictionary
        - initialize logging
        - create dictionary to save everything
    """
    
    assert hasattr(args, 'exp_name')

    args.summary_dir = os.path.join("experiments", args.exp_name, "summaries")
    args.checkpoint_dir = os.path.join("experiments", args.exp_name, "checkpoints")
    args.output_dir = os.path.join("experiments", args.exp_name, "output")
    args.log_dir = os.path.join("experiments", args.exp_name, "logs")

    Path(args.summary_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    init_logger(args.log_dir)


def init_deterministic():
    random_seed = 7
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) no multi-gpu supports for now
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
