import argparse
import logging
from modules.components import build_components
import random
import torch

from utils.config import parse_json
from utils.experiment import init_experiment, init_deterministic
from utils.misc import print_cuda_statistics

from dataloaders import build_data_loader
from modules.models import build_model
from trainer import Trainer


def main():
    arg_parser = argparse.ArgumentParser(description="Pytorch Project")
    arg_parser.add_argument(
        '--config',
        required=True,
        help='The configuration file in json format'
    )
    
    args = arg_parser.parse_args()
    json_args = parse_json(args.config)
    args = arg_parser.parse_args(namespace=json_args)

    init_experiment(args)

    device = init_device(args)
    train_loader, val_loader = build_data_loader(args)

    model = build_model(args, device)
    trainer = Trainer(model, train_loader, val_loader, args)

    assert hasattr(args, 'mode')
    if args.mode == 'training':
        trainer.train()
    elif args.mode == 'validation': # validation
        init_deterministic()
        trainer.validate()
    else:
        init_deterministic()
        trainer.test()

def init_device(args):
    # TODO: enable benchmark mode for more stable result
    logger = logging.getLogger("Main")
    is_cuda = torch.cuda.is_available()
    if is_cuda and not args.cuda:
        logger.info("You have CUDA device but the training does not use CUDA, you should probably enable CUDA")
    cuda = is_cuda & args.cuda
    manual_seed = args.seed
    
    logger.info("seed: " + str(manual_seed))
    if cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu_device)
        torch.cuda.manual_seed_all(manual_seed)
        logger.info("Program will run on GPU-CUDA on device number: {}".format(args.gpu_device))
        print_cuda_statistics()
    else:
        device = torch.device("cpu")
        torch.manual_seed(manual_seed)
        logger.info("Program will run on CPU")
    return device



if __name__ == '__main__':
    main()