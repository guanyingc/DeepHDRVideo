'''
Network architecture for refine net 
Deformable alignment HDR net (DAHDRnet)
Part of the codes are not used
'''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict
import models.archs.edvr_networks as edvr_net
from models import model_utils as mutils

class DAHDRnet(nn.Module):
    def __init__(self, c_in=6, c_out=3, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=10, other={}): 
        super(DAHDRnet, self).__init__()
        self.nf = nf = 64
        self.center = nframes // 2
        self.ref_skip = True
        self.other = other
        self.down_factor = 8

        ResidualBlock_noBN_f = functools.partial(edvr_net.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        self.conv_first_1 = nn.Conv2d(c_in, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = edvr_net.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        if self.other['PCD'] in ['', 'PCD']:
            self.pcd_align = edvr_net.PCD_Align(nf=nf, groups=groups)
        else:
            raise Exception('Unknown PCD module variant')

        if self.other['TAM'] == 'Conv':
            self.ta_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        elif self.other['TAM'] == 'ETA':
            self.ta_fusion = edvr_net.ETA_Fusion(nf=nf, nframes=nframes, center=self.center, in_mask=True)
        elif self.other['TAM'] == 'TA2': # use in this work
            self.ta_fusion = edvr_net.ETA_Fusion(nf=nf, nframes=nframes, center=self.center, in_mask=False)
        elif self.other['TAM'] in ['TA', '']:
            self.ta_fusion = edvr_net.TA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            raise Exception('Unknown TA module')

        #### reconstruction
        self.recon_trunk = edvr_net.make_layer(ResidualBlock_noBN_f, back_RBs)
        deconv_in = nf*2 if self.ref_skip else nf
        self.deconv1 = nn.ConvTranspose2d(deconv_in, nf, kernel_size=4, stride=2, padding=1, bias=True)
        
        self.HRconv = nn.Conv2d(deconv_in, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inputs):
        x = inputs['x']
        B, N, C, H, W = x.size()  # N video frames

        pad_img = False
        if not mutils.check_valid_input_size(self.down_factor, H, W):
            pad_img = True
            self.in_h, self.in_w = H, W
            #print('EDVR, resize imgs')
            #print(x.shape)
            x = mutils.pad_img_to_factor_of_k(x, k=self.down_factor)
            H, W = x.shape[-2:]
            #print(x.shape)

        x_center = x[:, self.center, :, :, :].contiguous()
        #### extract LR features
        # L1
        #if self.HR_in:
        low_feat1 = L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
        low_feat2 = L1_fea = self.lrelu(self.conv_first_2(L1_fea))
        H, W = H // 2, W // 2
        #    skipf3 = L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        #else:
        #    L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]

        if self.other['PCD'] == 'EPCD':
            exp_mask_l = self.get_multi_scale_exp_mask(inputs)

        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]

            if self.other['PCD'] == 'EPCD':
                aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l, exp_mask_l))
            else:
                aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))

        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if self.other['TAM'] in ['Conv']:
            aligned_fea = aligned_fea.view(B, -1, H, W)
            fea = self.ta_fusion(aligned_fea) # [B, C, H, W]
        elif self.other['TAM'] in ['ETA']:
            exp_mask = self.get_exp_mask(inputs)
            fea = self.ta_fusion(aligned_fea, exp_mask) # [B, C, H, W]
        else:
            fea = self.ta_fusion(aligned_fea) # [B, C, H, W]

        out = self.recon_trunk(fea)

        # concat here
        if self.ref_skip:
            ref_feat2 = low_feat2.view(B, N, -1, H, W)[:, self.center, :, :, :].clone()
            out = torch.cat([ref_feat2, out], 1)

        out = self.deconv1(out) # upsample
        if self.ref_skip:
            ref_feat1 = low_feat1.view(B, N, -1, H*2, W*2)[:, self.center, :, :, :].clone()
            out = torch.cat([ref_feat1, out], 1)

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)

        hdr = torch.sigmoid(out)
        if pad_img:
            hdr = hdr[:, :, :self.in_h, :self.in_w]
        pred = OrderedDict()
        pred['hdr'] = hdr
        return pred
    
    def get_multi_scale_exp_mask(self, inputs):
        exp_mask_l = []
        exp_mask_l.append(F.interpolate(inputs['gt_ref_w'], scale_factor=0.5, mode='bilinear', align_corners=False))
        exp_mask_l.append(F.interpolate(inputs['gt_ref_w'], scale_factor=0.25, mode='bilinear', align_corners=False))
        exp_mask_l.append(F.interpolate(inputs['gt_ref_w'], scale_factor=0.125, mode='bilinear', align_corners=False))
        return exp_mask_l

    def get_exp_mask(self, inputs):
        exp_mask = F.interpolate(inputs['gt_ref_w'], scale_factor=0.5, mode='bilinear', align_corners=False)
        return exp_mask

    def freeze_motion_layers(self, fuse_ta=True):
        self.motion_layers = ['conv_first_1', 'conv_first_2', 'feature_extraction']
        self.motion_layers += ['fea_L2_conv1', 'fea_L2_conv2']
        self.motion_layers += ['fea_L3_conv1', 'fea_L3_conv2']
        self.motion_layers += ['pcd_align']
        if fuse_ta:
            self.motion_layers += ['ta_fusion']
        
        for layer_name in self.motion_layers:
            print('Freezing %s' % layer_name)
            layer = getattr(self, layer_name)
            for param in layer.parameters():
                param.requires_grad = False
