"""
Util functions for model forward and backward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math
torch.manual_seed(0)
np.random.seed(0)

# Merge HDRs
def merge_hdr(ws, hdrs):
    assert(len(ws) == len(hdrs))
    w_sum = torch.stack(ws, 1).sum(1)
    ws = [w / (w_sum + 1e-8) for w in ws]

    hdr = ws[0] * hdrs[0]
    for i in range(1, len(ws)):
        hdr += ws[i] * hdrs[i]
    return hdr, ws

# Conversion between LDRs and HDRs
def pt_ldr_to_hdr(ldr, expo, gamma=2.2):
    #expo = epo.view(-1, 1, 1, 1)
    ldr = ldr.clamp(0, 1)
    ldr = torch.pow(ldr, gamma)
    hdr = ldr / expo
    return hdr

def pt_ldr_to_hdr_clamp(ldr, expo, gamma=2.2):
    #expo = epo.view(-1, 1, 1, 1)
    ldr = ldr.clamp(1e-8, 1)
    ldr = torch.pow(ldr, gamma)
    hdr = ldr / expo
    return hdr

def pt_hdr_to_ldr_clamp(hdr, expo, gamma=2.2,):
    ldr = torch.pow((hdr * expo).clamp(1e-8), 1.0 / gamma)
    ldr = ldr.clamp(0, 1)
    return ldr

def pt_hdr_to_ldr(hdr, expo, gamma=2.2):
    ldr = torch.pow(hdr * expo, 1.0 / gamma)
    ldr = ldr.clamp(0, 1)
    return ldr

def pt_ldr_to_ldr(ldr, expo_l2h, expo_h2l, gamma=2.2):
    ldr = ldr.clamp(0, 1)
    #if expo_l2h == expo_h2l:
    #    return ldr
    gain = torch.pow(expo_h2l / expo_l2h, 1.0 / gamma)
    ldr = (ldr * gain).clamp(0, 1)
    return ldr

# Generate mask 
def pt_get_out_blend_mask(ldr, h_idx, l_idx=None, m_idx=None, h_thr=0.9, l_thr=0.3, ue_pow=4, oe_pow=2):
    n, c, h, w = ldr.shape
    exp_mask = torch.zeros((n, 3, h, w), device=ldr.device)

    def get_over_expo_mask(img, h_thr, power=2):
        over_expo_mask = - ((img-h_thr)/(1-h_thr) - 1)**power + 1
        return over_expo_mask

    def get_under_expo_mask(img, l_thr, power=4):
        under_exp_mask = - (1 - (l_thr-img)/(l_thr-0))**power + 1
        return under_exp_mask

    if h_idx.sum() > 0:
        over_expo_region = (ldr[h_idx] > h_thr).float()
        exp_mask[h_idx] = over_expo_region * get_over_expo_mask(ldr[h_idx], h_thr, oe_pow)

    if l_idx is None:
        l_idx = 1 - h_idx
    if l_idx.sum() > 0:
        under_exp_region = (ldr[l_idx] < l_thr).float()
        exp_mask[l_idx] = under_exp_region * get_under_expo_mask(ldr[l_idx], l_thr, ue_pow)

    if m_idx is not None and m_idx.sum() > 0:
        #print('Get mid exposure mask')
        oe_region = (ldr[m_idx] > h_thr).float()
        ue_region = (ldr[m_idx] < l_thr).float()
        exp_mask[m_idx] = oe_region * get_over_expo_mask(ldr[m_idx], h_thr) + ue_region * get_under_expo_mask(ldr[m_idx], l_thr)
    return exp_mask

def pt_get_in_exposure_mask(ldr, h_idx, l_idx=None, m_idx=None, h_thr=0.9, l_thr=0.3):
    exp_mask = torch.zeros(ldr.shape, device=ldr.device)
    if h_idx.sum() > 0:
        exp_mask[h_idx] = (ldr[h_idx] > h_thr).float()

    if l_idx is None:
        l_idx = 1 - h_idx
    if l_idx.sum() > 0:
        exp_mask[l_idx] = (ldr[l_idx] < l_thr).float()

    if m_idx is not None and m_idx.sum() > 0:
        exp_mask[m_idx] = ((ldr[m_idx] > h_thr) | (ldr[m_idx] < l_thr)).float()
    return exp_mask

# Hard output mask
def pt_ldr_to_1c_mask(ldr, h_idx, h_thr=0.85, l_thr=0.35):
    n, c, h, w = ldr.shape
    exp_mask = torch.zeros((n, 1, h, w), device=ldr.device)
    if h_idx.sum() > 0:
        exp_mask[h_idx], _ = (ldr[h_idx] > h_thr).float().max(1, keepdim=True)
    if (1 - h_idx).sum() > 0:
        exp_mask[1-h_idx], _ = (ldr[1-h_idx] < l_thr).float().min(1, keepdim=True)
    return exp_mask

def pt_ldr_to_3exps_1c_mask(ldr, h_idx, m_idx, l_idx, h_thr=0.85, l_thr=0.35):
    n, c, h, w = ldr.shape
    exp_mask = torch.zeros((n, 1, h, w), device=ldr.device)
    if h_idx.sum() > 0:
        exp_mask[h_idx], _ = (ldr[h_idx] > h_thr).float().max(1, keepdim=True)
    if l_idx.sum() > 0:
        exp_mask[l_idx], _ = (ldr[l_idx] < l_thr).float().min(1, keepdim=True)
    if m_idx.sum() > 0:
        high_mask, _ = (ldr[m_idx] > h_thr).max(1, keepdim=True)
        low_mask, _ = (ldr[m_idx] < l_thr).min(1, keepdim=True)
        exp_mask[m_idx] = (high_mask | low_mask).float() 
    return exp_mask

# Training related
def get_lr_scheduler(opt, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=opt['milestones'], gamma=opt['lr_decay'], last_epoch=opt['start_epoch']-2)
    return scheduler

def init_weights(net, init_type='kaiming', init_gain=0.02):
    print('==> %s initilaization' % init_type)
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func) 

def init_net(net, init_type='kaiming', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type)
    torch.manual_seed(0)
    np.random.seed(0)
    return net

def get_params_num(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def get_image_stat(img, fmt='%.4f'):
    max_v, mean_v, min_v = img.max(), img.mean(), img.min()
    template = 'max: %s, mean: %s, min: %s' % (fmt, fmt, fmt)
    stat = template % (max_v, mean_v, min_v)
    return stat

def generate_grid(shape, device):
    n, c, h, w = shape
    hori_grid = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1).to(device)
    vert_grid = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w).to(device)
    print('Creating grid: %s' % (str(shape)))
    grid = torch.cat([hori_grid, vert_grid], 1) #[-1, 1]
    return grid #[hori_grid, vert_grid]

def backward_warp(img, flow, grid, pad='zeros'):
    n, c, h, w = img.shape
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], 1) # [-2, 2]
    accum_flow = grid + flow # [-1, 1]
    warped_img = torch.nn.functional.grid_sample(input=img, grid=accum_flow.permute(0, 2, 3, 1), mode='bilinear',  padding_mode='zeros') 
    return warped_img

def check_valid_input_size(factor_of_k, h, w):
    if h % factor_of_k != 0 or w % factor_of_k != 0:
        return False
    return True

def resize_flow(flow, h, w):
    b, c, c_h, c_w = flow.shape
    flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)
    flow[:, 0] *= float(w) / float(c_w) # rescale flow magnitude after rescaling spatial size
    flow[:, 1] *= float(h) / float(c_h)
    return flow

def resize_img_to_factor_of_k(img, k=64, mode='bilinear'):
    b, c, h, w = img.shape
    new_h = int(np.ceil(h / float(k)) * k)
    new_w = int(np.ceil(w / float(k)) * k)
    img = F.interpolate(input=img, size=(new_h, new_w), mode=mode, align_corners=False)
    return img

def pad_img_to_factor_of_k(img, k=64, mode='replicate'):
    if img.ndimension() == 4:
        b, c, h, w = img.shape
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
    elif img.ndimension() == 5:
        h, w = img.shape[-2], img.shape[-1]
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
    return img

