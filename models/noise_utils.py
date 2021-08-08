"""
Util functions for tone perturbation during training
"""
import torch
torch.manual_seed(0)

## Data augmentation used in this work
def pt_tone_ref_tone_augment(ldr, d=0.1):
    n, c, h, w = ldr.shape
    # d: [-0.7, 0.7] -> gamma [0.49, 2.0]
    gammas = torch.exp(torch.rand(n, 3, 1, 1) * 2 * d - d)
    gammas = gammas.to(ldr.device)
    ldr_tone_aug = torch.pow(ldr, 1.0 / gammas)
    return ldr_tone_aug

def pt_tone_ref_add_gaussian_noise(img, stdv1=1e-3, stdv2=1e-2, max_thres=0.08, scale=True):
    stdv = torch.rand(img.shape, device=img.device) * (stdv2 - stdv1) + stdv1
    noise = torch.normal(0, stdv)
    out = torch.pow(img.clamp(0, 1), 2.2) # undo gamma
    out = (out + noise).clamp(0, 1)
    out = torch.pow(out, 1/2.2)
    return out


