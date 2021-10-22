import torch
import torch.nn.functional as F

from torch import nn
from modules.initializer.weights_initializer import weights_init_kaiming
from .common import conv, ResidualBlock

class MaskNet(nn.Module):
    def __init__(self, input_nc, output_nc, base_features, kernel_size, bias=False):
        super(MaskNet, self).__init__()

        act = nn.PReLU()

        self.mask_net = nn.Sequential(
            conv(input_nc, base_features, kernel_size=kernel_size, bias=bias),
            ResidualBlock(base_features, kernel_size=kernel_size, bias=bias, act=act),
            ResidualBlock(base_features, kernel_size=kernel_size, bias=bias, act=act),
            ResidualBlock(base_features, kernel_size=kernel_size, bias=bias, act=act),
            conv(base_features, output_nc, kernel_size=kernel_size, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.mask_net(x)
        return out


if __name__ == "__main__":
    net = MaskNet(3, 3, 32, 3, bias=False).cuda()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(4, 3, 256, 256).cuda()
    out = net(x)
    print(out.size())
