import torch
import torch.nn as nn
from models import model_utils as mutils
from models import network_utils as nutils
from collections import OrderedDict

class Downsample(nn.Module):
    def __init__(self, c_in, c_out=512, use_bn=True, afunc='LReLU'):
        super(Downsample, self).__init__()

        self.layers = torch.nn.Sequential(
            nutils.conv_layer(c_in,     c_out//4, k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn), 
            nutils.conv_layer(c_out//4, c_out//4, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            nutils.conv_layer(c_out//4, c_out//2, k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn),
            nutils.conv_layer(c_out//2, c_out//2, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            nutils.conv_layer(c_out//2, c_out,    k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn), 
            nutils.conv_layer(c_out,    c_out,    k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
        )

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

class Upsample(nn.Module):
    def __init__(self, c_in, c_out, use_bn=True, afunc='LReLU'):
        super(Upsample, self).__init__()
        last_c = max(64, c_in // 8)
        self.layers = torch.nn.Sequential(
            nutils.deconv_layer(c_in, c_in//2, use_bn=use_bn, afunc=afunc),
            nutils.conv_layer(c_in//2, c_in//2, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            nutils.deconv_layer(c_in//2,  c_in//4, use_bn=use_bn, afunc=afunc),
            nutils.conv_layer(c_in//4, c_in//4, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            nutils.deconv_layer(c_in//4, last_c, use_bn=use_bn, afunc=afunc),
            nutils.output_conv(last_c, c_out, k=3, stride=1, pad=1),
        )

    def forward(self, inputs):
        out = self.layers(inputs)
        out = torch.sigmoid(out)
        return out

class weight_EG19net(nn.Module):
    def __init__(self, c_in, c_out, c_mid=512, use_bn=False, afunc='LReLU', other={}):
        super(weight_EG19net, self).__init__()
        self.downsample = Downsample(c_in=c_in, c_out=c_mid, use_bn=use_bn, afunc=afunc)
        self.upsample = Upsample(c_in=c_mid, c_out=c_out, use_bn=use_bn, afunc=afunc)
        self.merge_HDR = nutils.MergeHDRModule()

        self.down_factor = 8

    def forward(self, inputs, hdrs):
        pred = OrderedDict()

        n, c, in_h, in_w = inputs.shape 
        pad_img = False
        if not mutils.check_valid_input_size(self.down_factor, in_h, in_w):
            #print('Weight net: Resizing imgs')
            inputs = mutils.pad_img_to_factor_of_k(inputs, k=self.down_factor)
            pad_img = True

        down_feature = self.downsample(inputs) 
        weights = self.upsample(down_feature)

        if pad_img:
            weights = weights[:, :, :in_h, :in_w]
        ws = torch.split(weights, 3, 1)
        
        hdr, weights = self.merge_HDR(ws, hdrs)
        pred['weights'] = weights
        pred['hdr'] = hdr
        return pred
