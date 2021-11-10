import torch

from torch import nn

class TotalVariationLoss(nn.Module):
    def forward(self, img) -> torch.Tensor:
        pixel_diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
        pixel_diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]

        reduce_axes = (-3, -2, -1)
        res_h = pixel_diff_h.abs().sum(dim=reduce_axes)
        res_w = pixel_diff_w.abs().sum(dim=reduce_axes)

        return res_h + res_w
