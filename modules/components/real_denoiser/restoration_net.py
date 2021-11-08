import torch
import torch.nn.functional as F

from torch import nn
from .common import conv, Encoder, Decoder, ResidualBlock

class RestorationNetV0(nn.Module):
    def __init__(self, input_nc, output_nc, base_features, scale_features, kernel_size, bias=False):
        super(RestorationNetV0, self).__init__()

        act = nn.ReLU()

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
        self.pre_aux_head = conv(base_features, base_features, kernel_size, bias=bias)
        self.aux_pri_conjunction = conv(base_features + output_nc, base_features, kernel_size, bias=bias)

        # Auxiliary Head
        self.auxiliary_head = nn.Sequential(
            ResidualBlock(base_features, kernel_size, bias, act),
            conv(base_features, output_nc, kernel_size, bias=bias)
        )

    def forward(self, x, weights=None):
        feat = self.shallow_fe(x)
        feat = self.encoder(feat)
        feat = self.decoder(feat)
        
        feat = feat[0]
        residual = self.primary_head(feat)
        clean_pred = residual + x

        feat = self.pre_aux_head(feat)
        feat = torch.cat([feat, residual], dim=1)
        feat = self.aux_pri_conjunction(feat)
        reconstructed_noisy = self.auxiliary_head(feat)

        return clean_pred, reconstructed_noisy

class RestorationNet(nn.Module):
    def __init__(self, input_nc, output_nc, base_features, scale_features, kernel_size, bias=False):
        super(RestorationNet, self).__init__()

        act = nn.ReLU()

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

    def forward(self, x, weights=None):
        if weights is None:
            feat = self.shallow_fe(x)
            feat = self.encoder(feat)
            feat = self.decoder(feat)
            
            feat = feat[0]
            residual = self.primary_head(feat)
            clean_pred = residual + x

            reconstructed_noisy = self.auxiliary_head(torch.cat([clean_pred, residual], dim=1))
        else:
            feat = shallow_fe_ff(x, weights)
            feat = encoder_ff(feat, weights)
            feat = decoder_ff(feat, weights)

            feat = feat[0]
            residual = primary_head_ff(feat, weights)
            clean_pred = residual + x
            
            reconstructed_noisy = auxiliary_head_ff(torch.cat([clean_pred, residual], dim=1), weights)

        return clean_pred, reconstructed_noisy

def shallow_fe_ff(x, weights):
    net = F.conv2d(x, weights['shallow_fe.0.weight'], padding=1)
    return net

def encoder_ff(x, weights):
    # Encoder L1
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l1.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l1.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l1.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l1.1.model.2.weight'], padding=1)
    net = net + x
    ##
    enc_out1 = net

    # Down 12 Block
    net = F.interpolate(net, scale_factor=0.5, mode='bilinear', align_corners=False)
    net = F.conv2d(net, weights['encoder.down12.conv.weight'], padding=0)

    x = net
    # Encoder L2
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l2.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l2.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l2.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l2.1.model.2.weight'], padding=1)
    net = net + x
    ##
    enc_out2 = net

    # Down 23 Block
    net = F.interpolate(net, scale_factor=0.5, mode='bilinear', align_corners=False)
    net = F.conv2d(net, weights['encoder.down23.conv.weight'], padding=0)

    x = net
    # Encoder L3
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l3.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l3.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['encoder.encoder_l3.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['encoder.encoder_l3.1.model.2.weight'], padding=1)
    net = net + x
    ##
    enc_out3 = net
    return [enc_out1, enc_out2, enc_out3]

def decoder_ff(x, weights):
    enc_out1, enc_out2, enc_out3 = x

    # Decoder L3
    x = enc_out3
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l3.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l3.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l3.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l3.1.model.2.weight'], padding=1)
    net = net + x
    ##
    dec_out3 = net

    ## Upsampling 32 Block
    net = F.interpolate(net, size=enc_out2.size()[2:], mode='bilinear', align_corners=False)
    net = F.conv2d(net, weights['decoder.up32.conv.weight'], padding=0) + enc_out2

    x = net
    # Decoder L2
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l2.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l2.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l2.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l2.1.model.2.weight'], padding=1)
    net = net + x
    ##
    dec_out2 = net

    ## Upsampling 21 Block
    net = F.interpolate(net, size=enc_out1.size()[2:], mode='bilinear', align_corners=False)
    net = F.conv2d(net, weights['decoder.up21.conv.weight'], padding=0) + enc_out1

    x = net
    # Decoder L1
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l1.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l1.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['decoder.decoder_l1.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['decoder.decoder_l1.1.model.2.weight'], padding=1)
    net = net + x
    ##
    dec_out1 = net

    return [dec_out1, dec_out2, dec_out3]

def primary_head_ff(x, weights):
    ## ResidualBlock
    net = F.conv2d(x, weights['primary_head.0.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['primary_head.0.model.2.weight'], padding=1)
    net = net + x
    ##
    x = net
    ## ResidualBlock
    net = F.conv2d(x, weights['primary_head.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['primary_head.1.model.2.weight'], padding=1)
    net = net + x
    ##
    net = F.conv2d(net, weights['primary_head.2.weight'], padding=1)
    return net

def auxiliary_head_ff(x, weights):
    x = F.conv2d(x, weights['auxiliary_head.0.weight'], padding=1)
    ## ResidualBlock
    net = F.conv2d(x, weights['auxiliary_head.1.model.0.weight'], padding=1)
    net = F.relu(net)
    net = F.conv2d(net, weights['auxiliary_head.1.model.2.weight'], padding=1)
    net = net + x
    ##
    net = F.conv2d(net, weights['auxiliary_head.2.weight'], padding=1)
    return net

if __name__ == "__main__":
    from modules.initializer.weights_initializer import weights_init_kaiming
    net = RestorationNet(3, 3, 32, 16, kernel_size=3, bias=False).cuda()
    net.apply(weights_init_kaiming)
    print(net)
    for name, param in net.named_parameters():
        print(name)
    x = torch.randn(4, 3, 256, 256).cuda()
    out = net(x)

    # Shallow FE Test
    x = torch.randn(4, 3, 64, 64).cuda()
    out_shallow_fe = net.shallow_fe(x)
    out_shallow_fe_ff = shallow_fe_ff(x, net.state_dict())
    assert torch.all(torch.eq(out_shallow_fe, out_shallow_fe_ff))

    # Encoder Test
    x = torch.randn(4, 32, 64, 64).cuda()
    out_encoder = net.encoder(x)
    out_encoder_ff = encoder_ff(x, net.state_dict())
    assert torch.all(torch.eq(out_encoder[0], out_encoder_ff[0]))
    assert torch.all(torch.eq(out_encoder[1], out_encoder_ff[1]))
    assert torch.all(torch.eq(out_encoder[2], out_encoder_ff[2]))

    # Decoder Test
    x = [torch.randn(4, 32, 64, 64).cuda(), torch.randn(4, 48, 32, 32).cuda(), torch.randn(4, 64, 16, 16).cuda()]
    out_decoder = net.decoder(x)
    out_decoder_ff = decoder_ff(x, net.state_dict())
    assert torch.all(torch.eq(out_decoder[2], out_decoder_ff[2]))
    assert torch.all(torch.eq(out_decoder[1], out_decoder_ff[1]))
    assert torch.all(torch.eq(out_decoder[0], out_decoder_ff[0]))

    # Prim Head Test
    x = torch.randn(4, 32, 64, 64).cuda()
    out_prim_head = net.primary_head(x)

    out_prim_head_ff = primary_head_ff(x, net.state_dict())
    assert torch.all(torch.eq(out_prim_head, out_prim_head_ff))

    # Aux Head Test
    x = torch.randn(4, 6, 32, 32).cuda()    
    out_aux_head = net.auxiliary_head(x)

    out_aux_head_ff = auxiliary_head_ff(x, net.state_dict())
    assert torch.all(torch.eq(out_aux_head, out_aux_head_ff))
    
    # All Combined
    x = torch.randn(4, 3, 256, 256).cuda()
    out_net = net(x)
    out_net_ff = net(x, net.state_dict())

    assert torch.all(torch.eq(out_net[0], out_net_ff[0]))
    assert torch.all(torch.eq(out_net[1], out_net_ff[1]))