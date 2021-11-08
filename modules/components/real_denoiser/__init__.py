from torch import nn
from .mask_net import MaskNet
from .restoration_net import RestorationNetV0

def build_components(args):
    restoration_criterion = nn.L1Loss()

    mask_net = MaskNet(args.input_channels, 1, base_features=args.mask_base_features, kernel_size=3, bias=False)
    restoration_net = RestorationNetV0(args.input_channels, args.output_channels, args.restoration_base_features, args.restoration_scale_features, kernel_size=3, bias=False)
    return mask_net, restoration_net, restoration_criterion
