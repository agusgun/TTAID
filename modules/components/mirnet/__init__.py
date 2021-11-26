from .MIRNet_model import MIRNet
from .losses import CharbonnierLoss

def build_components(args):
    restoration_criterion = CharbonnierLoss()

    restoration_net = MIRNet()
    return restoration_net, restoration_criterion
