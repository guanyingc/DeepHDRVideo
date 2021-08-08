import torch
import math
import numpy as np
from matplotlib import cm
import matplotlib
import torch.nn.functional as F
import cv2
np.random.seed(0)

from extensions.pytorch_msssim import ssim_matlab as ssim_pth

def pt_quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()

def pt_calc_psnr(pred, gt, mask=None):
    # Assume input in the range of [0, 1]
    diff = pred - gt
    #diff = (pred - gt).div(255)
    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() 
        if mse == 0:
            mse += 1e-8

    return -10 * math.log10(mse)

def calc_ssim(img1, img2, datarange=255.):
    im1 = img1.numpy().transpose(1, 2, 0).astype(np.uint8)
    im2 = img2.numpy().transpose(1, 2, 0).astype(np.uint8)
    return compare_ssim(im1, im2, datarange=datarange, multichannel=True, gaussian_weights=True)

def pt_calc_metrics(im_pred, im_gt, mask=None): # Swap pred and gt
    #q_im_pred = pt_quantize(im_pred.data, rgb_range=1.)
    #q_im_gt = pt_quantize(im_gt.data, rgb_range=1.)
    q_im_pred = im_pred # * 255.0
    q_im_gt = im_gt # * 255.0
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = pt_calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())
    ssim = ssim_pth(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), val_range=1.0)
    return psnr, ssim

def pt_eval_metrics(output, gt, mask=None, psnrs_masked=None, ssims_masked=None):
    # PSNR should be calculated for each image
    B = gt.shape[0]
    psnrs = torch.zeros(B, device=output.device)
    ssims = torch.zeros(B, device=output.device)
    for b in range(B):
        if mask is None:
            psnr, ssim = pt_calc_metrics(output[b], gt[b], None)
        else:
            psnr, ssim= pt_calc_metrics(output[b], gt[b], mask[b])
        psnrs[b] = psnr
        ssims[b] = ssim
    return psnrs, ssims

def pt_blend_images(imgs):
    blend_img = imgs[0].clone()
    for i in range(1, len(imgs)):
        blend_img += imgs[i]
    return blend_img / len(imgs)

def pt_flow_to_color(flow, anchor=150):
    f_dx = flow.narrow(1, 0, 1) # Nx1xHxW
    f_dy = flow.narrow(1, 1, 1)
    f_mag = torch.sqrt(torch.pow(f_dx, 2) + torch.pow(f_dy, 2))
    f_dir = torch.atan2(f_dy, f_dx)
    f_dir = (f_dir + np.pi) / (2 * np.pi) # from [-pi, pi] to [0, 1] 

    n, c, h, w = flow.shape
    flow_hsv = f_dir.new_zeros(n, 3, h, w)
    flow_hsv.narrow(1, 0, 1).copy_(f_dir)
    flow_hsv.narrow(1, 1, 1).copy_((f_mag / anchor).clamp(0, 1)) # adjust scale
    #flow_hsv.narrow(1, 1, 1).copy_((f_mag / (f_mag.shape[3] * scale)).clamp(0, 1)) # adjust scale
    flow_hsv[:, 2] = 1
    flow_rgb = torch.from_numpy(matplotlib.colors.hsv_to_rgb(flow_hsv.permute(0, 2, 3, 1).cpu().numpy()))
    flow_rgb = flow_rgb.permute(0, 3, 1, 2)
    return flow_rgb

def pt_mulog_transform(in_tensor, mu=5000.0):
    denom = torch.log(in_tensor.new([1.0 + mu]))
    out_tensor = torch.log(1.0 + mu * in_tensor) / denom 
    return out_tensor

def pt_inverse_mulog_transform(in_tensor, mu=5000.0):
    out_tensor = (torch.exp(in_tensor * torch.log(in_tensor.new([1 + mu]))) - 1.0)  / mu
    return out_tensor

def pt_colormap(diff, thres=0.1):
    diff_norm = diff.clamp(0, thres) / thres
    diff_cm = torch.from_numpy(cm.jet(diff_norm.cpu().numpy()))[:,:,:, :3]
    return diff_cm.permute(0,3,1,2).clone().float()

def pt_cal_psnr_batch(pred, target): # n x PSNR
    mse = torch.pow(pred - target, 2).view(pred.shape[0], -1).mean(1)
    zero_mask = (mse == 0.)
    mse[zero_mask] = 1
    PIXEL_MAX = 1.0
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    psnr[zero_mask] = 100
    return psnr, mse

def pt_cal_psnr(pred, target):
    mse = pt_cal_msr(pred, target)
    if mse == 0:
        return 100, 100
    PIXEL_MAX = 1.0
    psnr = 20 * np.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr, mse

def pt_cal_msr(pred, target):
    error = torch.pow(pred - target, 2).mean()
    return error.item()

def pt_cal_diff_map(pred, target):
    diff = torch.sqrt(torch.pow(pred - target, 2).mean(1))
    return diff

def crop_tensor(tensor, h, w):
    n, c, th, tw = tensor.shape
    if th > h or tw > w:
        tensor = tensor[:, :, :h, :w]
    return tensor

def crop_list_of_tensors(list_of_tensors, hw):
    new_list = []
    for tensor in list_of_tensors:
        new_list.append(crop_tensor(tensor, hw[0], hw[1]))
        #n, c, h, w = tensors.shape
        #if h >= hw[0] and w >= hw[1]:
        #    new_list.append(tensors[:, :, :hw[0], :hw[1]])
    return new_list
 
# Optical flow related
def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        weights = weights[::-1]
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def pt_enhance_loghdr(pt_img, saturation_fac=1.6):
    device = pt_img.device
    B, C, H, W = pt_img.shape
    img = pt_img.cpu().float().permute(0, 2, 3, 1).numpy()
    new_img = np.zeros(img.shape, dtype=img.dtype)
    for b in range(B):
        hsv_i = cv2.cvtColor(img[b], cv2.COLOR_RGB2HSV)
        hsv_i[:,:,1] = hsv_i[:,:,1] * saturation_fac
        new_img[b] = cv2.cvtColor(hsv_i, cv2.COLOR_HSV2RGB)
    new_img = torch.from_numpy(new_img).permute(0, 3, 1, 2).to(device)
    return new_img
