"""
Util functions for data loading and processing
"""
import torch
import random
import numpy as np
from skimage.transform import resize, rotate
import cv2
random.seed(0)
np.random.seed(0)

def array_to_tensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()


def imgsize_to_factor_of_k(img, k):
    if img.shape[0] % k == 0 and img.shape[1] % k == 0:
        return img
    pad_h, pad_w = k - img.shape[0] % k, k - img.shape[1] % k
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), 'reflect')
    #img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), 'edge')
            #'constant', constant_values=((0,0),(0,0),(0,0)))
    return img


def random_crop(inputs, size, margin=0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img in inputs:
            outputs.append(img[t: t+c_h, l: l+c_w])
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs 


def rescale(inputs, size):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    if h != size[0] or w != size[1]:
        for img in inputs:
            #out_img = resize(img, size, order=1, mode='reflect')
            out_img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            outputs.append(out_img)
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs


def pad_image(inputs, h=128, w=128):
    new_shape = (h, w) if inputs.ndim == 2 else (h, w, inputs.shape[2])
    outputs = np.zeros(new_shape, dtype=inputs.dtype)

    in_h, in_w = inputs.shape[0], inputs.shape[1]
    pad_h, pad_w = (h - in_h) // 2, (w - in_w) // 2
    t, b, l, r = pad_h, pad_h + in_h, pad_w, pad_w + in_w
    outputs[t:b, l:r] = inputs 
    return outputs


def random_noise_aug(inputs, noise_level=0.05):
    if not __debug__: print('RandomNoiseAug: input, noise level', inputs.shape, noise_level)
    noise = np.random.random(inputs.shape)
    noise = (noise - 0.5) * noise_level
    inputs += noise
    return inputs


def add_gaussian_noise(inputs, mean=0, stdv=[1e-3, 3e-3], per_channel=False):
    shape = inputs.shape
    stdv = np.random.uniform(stdv[0], stdv[1])
    noise = np.random.normal(loc=mean, scale=stdv, size=shape)
    inputs += noise
    return inputs.clip(0, 1)


def random_flip(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    for img in inputs:
        outputs.append(np.fliplr(img).copy())
    if not is_list: outputs = outputs[0]
    return outputs


def random_flip_lrud(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img in inputs:
        flip_img = np.fliplr(img)
        if vertical_flip:
            flip_img = np.flipud(flip_img)
        outputs.append(flip_img.copy())
    if not is_list: outputs = outputs[0]
    return outputs


def random_rotate90(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    for img in inputs:
        outputs.append(rotate(img, angle=90, mode='constant'))
    if not is_list: outputs = outputs[0]
    return outputs


def random_rotate(inputs, angle=90.0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    ang = np.random.uniform(0, angle)
    for img in inputs:
        outputs.append(rotate(img, angle=ang, mode='constant'))
    if not is_list: outputs = outputs[0]
    return outputs


def apply_gamma(img, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, 1.0 / gamma)
    return img


def undo_gamma(img, gamma=2.2):
    img = img.clip(0, 1)
    img = np.power(img, gamma)
    return img


def translate(img, offsets):
    disp_w, disp_h = offsets
    T = np.float32([[1, 0, disp_w], [0, 1, disp_h]]) 
    h, w, _ = img.shape
    img_translated = cv2.warpAffine(img, T, (w, h)) 
    return img_translated


def random_translate(img, max_disp=5):
    disp_w = np.random.uniform(-max_disp, max_disp+1)
    disp_h = np.random.uniform(-max_disp, max_disp+1)
    #disp_w, disp_h = 40, 40 # TODO
    img_translated = translate(img, [disp_w, disp_h])
    return img_translated


def crop_border(img, border=20):
    img = img[border: -border, border: -border]
    return img


def center_crop(imgs, crop_h, crop_w):
    new_imgs = []
    for img in imgs:
        h, w, c = img.shape
        t, l = (h - crop_h) // 2, (w - crop_w) // 2
        b, r = t + crop_h, l + crop_w
        new_imgs.append(img[t:b, l:r])
    return new_imgs


# For online global alignment
def cvt_MToTheta(M, w, h):
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)

