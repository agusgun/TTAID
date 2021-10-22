import math

from torch import nn
from torch.nn import init

# Note: never access .data, it can produce unwawnted behaviors, see PyTorch forum

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# def weights_init_normal(m):
#     """
#     Initialize the weights of Convolution2D and BatchNorm2D with normal.
#     :param m:
#     :return:
#     """
#     if isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(0.0, 0.02)
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# def init_model_weights(m):
#     ### initialize
#     for m in m.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.bias.data.zero_()

# def weights_init_kaiming(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
#         nn.init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)