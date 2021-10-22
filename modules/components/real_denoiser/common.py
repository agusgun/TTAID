from torch import nn
import torch
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding, padding_size, norm_layer, act_func=nn.ReLU(inplace=True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding, padding_size, norm_layer, act_func, use_dropout)

    def build_conv_block(self, dim, padding, padding_size, norm_layer, act_func, use_dropout):
        conv_block = []
        conv_block += [
            padding(padding_size),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            act_func,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
            padding(padding_size),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                padding=nn.ZeroPad2d, padding_size=0, norm=nn.BatchNorm2d, act_func=nn.ReLU(inplace=True), bias=True):
        super(Conv2dBlock, self).__init__()
        self.model = [
            padding(padding_size),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias),
            norm(out_channels),
            act_func
        ]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        out = self.model(x)
        return out

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                padding=nn.ZeroPad2d, padding_size=0, norm=nn.BatchNorm2d, act_func=nn.ReLU(inplace=True), bias=True):
        super(Deconv2dBlock, self).__init__()
        self.model = [
            padding(padding_size),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias),
            norm(out_channels),
            act_func
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, target_h, target_w):
        out = F.interpolate(x, size=(target_h, target_w), mode='nearest')
        out = self.model(out)
        return out

######### New standardized network without any normalization layer from MPRNet (swz30/MPRNet) #########

def conv(in_channels, out_channels, kernel_size, stride=1, bias=False, padding=None):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=(kernel_size//2), bias=bias)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DownsamplingBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv = conv(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = self.conv(x)
        return out

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.conv = conv(in_channels, out_channels, 1)

    def forward(self, x, y):
        _, _, target_h, target_w = y.size()
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        out = self.conv(x)
        return out

class SkipUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipUpsample, self).__init__()
        self.conv = conv(in_channels, out_channels, 1)
    
    def forward(self, x, y):
        _, _, target_h, target_w = y.size()
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = self.conv(x)
        out = x + y
        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel_size, bias, act):
        super(ResidualBlock, self).__init__()
        modules = [
            conv(dim, dim, kernel_size, bias=bias),
            act,
            conv(dim, dim, kernel_size, bias=bias)
        ]
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        out = self.model(x)
        out = out + x
        return out


# U-Net
class Encoder(nn.Module):
    def __init__(self, base_features, scale_features, kernel_size, bias, act):
        super(Encoder, self).__init__()

        self.encoder_l1 = [ResidualBlock(base_features, kernel_size, bias, act) for _ in range(2)]
        self.encoder_l2 = [ResidualBlock(base_features + scale_features, kernel_size, bias, act) for _ in range(2)]
        self.encoder_l3 = [ResidualBlock(base_features + scale_features * 2, kernel_size, bias, act) for _ in range(2)]

        self.encoder_l1 = nn.Sequential(*self.encoder_l1)
        self.encoder_l2 = nn.Sequential(*self.encoder_l2)
        self.encoder_l3 = nn.Sequential(*self.encoder_l3)

        self.down12 = DownsamplingBlock(base_features, base_features + scale_features, 0.5)
        self.down23 = DownsamplingBlock(base_features + scale_features, base_features + scale_features * 2, 0.5)

    def forward(self, x):
        enc_out1 = self.encoder_l1(x)

        x = self.down12(enc_out1)
        enc_out2 = self.encoder_l2(x)

        x = self.down23(enc_out2)
        enc_out3 = self.encoder_l3(x)

        return [enc_out1, enc_out2, enc_out3]

class Decoder(nn.Module):
    def __init__(self, base_features, scale_features, kernel_size, bias, act):
        super(Decoder, self).__init__()

        self.decoder_l1 = [ResidualBlock(base_features, kernel_size, bias, act) for _ in range(2)]
        self.decoder_l2 = [ResidualBlock(base_features + scale_features, kernel_size, bias, act) for _ in range(2)]
        self.decoder_l3 = [ResidualBlock(base_features + scale_features * 2, kernel_size, bias, act) for _ in range(2)]

        self.decoder_l1 = nn.Sequential(*self.decoder_l1)
        self.decoder_l2 = nn.Sequential(*self.decoder_l2)
        self.decoder_l3 = nn.Sequential(*self.decoder_l3)

        self.up32 = SkipUpsample(base_features + scale_features * 2, base_features + scale_features)
        self.up21 = SkipUpsample(base_features + scale_features, base_features)

    def forward(self, enc_outs):
        enc_out1, enc_out2, enc_out3 = enc_outs
        dec_out3 = self.decoder_l3(enc_out3)

        x = self.up32(dec_out3, enc_out2)
        dec_out2 = self.decoder_l2(x)
        
        x = self.up21(dec_out2, enc_out1)
        dec_out1 = self.decoder_l1(x)
        
        return [dec_out1, dec_out2, dec_out3]

