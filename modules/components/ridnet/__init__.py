from torch import nn
from .ridnet import RIDNET

def build_components(args):
    restoration_criterion = nn.L1Loss()

    restoration_net = RIDNET(args)
    return restoration_net, restoration_criterion
