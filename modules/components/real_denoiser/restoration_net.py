import torch
import torch.nn.functional as F

from torch import nn
from .common import conv, Encoder, Decoder, ResidualBlock

class RestorationNet(nn.Module):
    def __init__(self, input_nc, output_nc, base_features, scale_features, kernel_size, bias=False):
        super(RestorationNet, self).__init__()

        act = nn.PReLU()

        # Network Body
        self.shallow_fe = nn.Sequential(
            conv(input_nc, base_features, kernel_size, bias=bias)
        )

        self.encoder = Encoder(base_features, scale_features, kernel_size, bias, act)
        self.decoder = Decoder(base_features, scale_features, kernel_size, bias, act)

        # Primary Head
        self.primary_head = nn.Sequential(
            ResidualBlock(base_features, kernel_size, bias, act),
            ResidualBlock(base_features, kernel_size, bias, act),
            conv(base_features, output_nc, kernel_size, bias=bias)
        )
        
        # Auxiliary Head
        self.auxiliary_head = nn.Sequential(
            conv(output_nc * 2, base_features, kernel_size, bias=bias),
            ResidualBlock(base_features, kernel_size, bias, act),
            conv(base_features, output_nc, kernel_size, bias=bias)
        )

    def forward(self, x):
        feat = self.shallow_fe(x)
        feat = self.encoder(feat)
        feat = self.decoder(feat)
        
        feat = feat[0]
        residual = self.primary_head(feat)
        clean_pred = residual + x

        reconstructed_noisy = self.auxiliary_head(torch.cat([clean_pred, residual], dim=1))

        return clean_pred, reconstructed_noisy


if __name__ == "__main__":
    net = RestorationNet(3, 3, 32, 16, kernel_size=3, bias=False).cuda()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(4, 3, 256, 256).cuda()
    out = net(x)
    print(out.size())
