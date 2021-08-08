"""
Dataloader for processing vimeo videos into training data
"""
import os
import numpy as np
from imageio import imread

import torch
import torch.utils.data as data

from datasets import hdr_transforms
from utils import utils
from utils import image_utils as iutils
np.random.seed(0)

class syn_vimeo_dataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(milestones=[5, 10, 20])
        return parser

    def __init__(self, args, split='train'):
        self.root = args.data_dir
        self.split = split
        self.args = args
        self.nframes = args.nframes
        self.nexps = args.nexps
        self._prepare_list(split)
        self._config_mode(split)

    def _prepare_list(self, split='train'):
        if split == 'train':
            list_name = 'sep_trainlist.txt'
        elif split == 'val':
            list_name = 'sep_testlist.txt'

        self.patch_list = utils.read_list(os.path.join(self.root, list_name))

        if split == 'val':
            self.patch_list = self.patch_list[:100] # only use 100 val patches

    def _config_mode(self, split='train'):
        # TODO: clean
        self.l_suffix = ''
        if split == 'train' and self.args.split_train == 'high_only':
            self.l_suffix = '_high.txt'
        elif split == 'train' and self.args.split_train == 'low_only':
            self.l_suffix = '_low.txt'
        elif self.args.split_train not in ['', 'high_only', 'low_only']:
            raise Exception('Unknown split type')

        if self.nexps == 2:
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat
        else:
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat

    def __getitem__(self, index):
        #index = 0
        img_paths, min_percent, exposures = self._get_input_path(index)
        hdrs = []

        """ sample parameters for the camera curves"""
        n, sigma = self.sample_camera_curve() # print(n, sigma)

        for img_path in img_paths:
            img = (imread(img_path).astype(np.float32) / 255.0).clip(0, 1)

            """ convert the LDR images to linear HDR image"""
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        h, w, c = hdrs[0].shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w

        if self.args.rescale and not (crop_h == h):
            max_h = h * self.args.sc_k
            max_w = w * self.args.sc_k
            sc_h = np.random.randint(crop_h, max_h+1) if self.args.rand_sc else self.args.scale_h
            sc_w = np.random.randint(crop_w, max_w+1) if self.args.rand_sc else self.args.scale_w
            hdrs = hdr_transforms.rescale(hdrs, [sc_h, sc_w])

        if self.args.flip_aug:
            hdrs = hdr_transforms.random_flip_lrud(hdrs)

        if self.args.rotate_aug:
            hdrs = hdr_transforms.random_rotate(hdrs, angle=10.0)

        if self.args.crop:
            hdrs = hdr_transforms.random_crop(hdrs, [crop_h, crop_w])

        if self.args.cl_aug:
            color_permute = np.random.permutation(3)
            for i in range(len(hdrs)):
                hdrs[i] = hdrs[i][:,:,color_permute]

        item = {}
        hdrs, ldrs, anchor = self.re_expose_ldrs(hdrs, min_percent, exposures)

        for i in range(1, self.nframes - 1):
            item['hdr_%d' % i] = hdrs[i-1]

        for i in range(len(ldrs)):
            item['ldr_%d' % i] = ldrs[i]

        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])

        item['expos'] = exposures
        return item

    def _get_input_path(self, index, high_expo=8):
        scene, patch = self.patch_list[index // self.repeat].split('/')
        img_dir = os.path.join(self.root, 'sequences', self.patch_list[index // self.repeat])

        img_idxs = sorted(np.random.permutation(7)[:self.nframes] + 1)
        if self.args.inv_aug and np.random.random() > 0.5: # inverse time order
            img_idxs = img_idxs[::-1]

        img_paths = [os.path.join(img_dir, 'im%d.png' % idx) for idx in img_idxs]
        min_percent = 99.8
        if self.nexps == 2:
            exposures = self._get_2exposures(index)
        elif self.nexps == 3:
            exposures = self._get_3exposures(index)
        else:
            raise Exception("Unknow exposures")
        return img_paths, min_percent, exposures

    def sample_camera_curve(self):
        n = np.clip(np.random.normal(0.65, 0.1), 0.4, 0.9)
        sigma = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.8)

        #n = np.clip(np.random.normal(0.9, 0.1), 0.7, 1.2)
        #sigma = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.8)
        return n, sigma

    def apply_sigmoid_curve(self, x, n, sigma):
        y = (1 + sigma) * np.power(x, n) / (np.power(x, n) + sigma)
        return y

    def apply_inv_sigmoid_curve(self, y, n, sigma):
        x = np.power((sigma * y) / (1 + sigma - y), 1/n)
        return x

    def apply_inv_s_curve(self, y):
        x = 0.5 - np.sin(np.arcsin(1 - 2*y)/3.0)
        return x

    def discretize_to_uint16(self, img):
        max_int = 2**16-1
        img_uint16 = np.uint16(img * max_int).astype(np.float) / max_int
        return img_uint16

    def _get_2exposures(self, index):
        cur_high = True if index % 2 == 0 else False

        # TODO:clean
        if self.l_suffix != '':
            cur_high = True if self.l_suffix == '_high.txt' else False

        exposures = np.ones(self.nframes, dtype=np.float32)
        high_expo = np.random.choice([4., 8.]) if self.args.rstop else 8

        if cur_high:
            for i in range(0, self.nframes, 2):
                exposures[i] = high_expo
        else:
            for i in range(1, self.nframes, 2):
                exposures[i] = high_expo
        return exposures

    def _get_3exposures(self, index):
        if index % self.nexps == 0:
            exp1 = 1
        elif index % self.nexps == 1:
            exp1 = 4
        else:
            exp1 = 16
        expos = [exp1]
        for i in range(1, self.nframes):
            if expos[-1] == 1:
                expos.append(4)
            elif expos[-1] == 4:
                expos.append(16)
            elif expos[-1] == 16:
                expos.append(1)
            else:
                raise Exception('Unknown expos %d' % expos[-1])
        exposures = np.array(expos).astype(np.float32)
        return exposures

    def re_expose_ldrs(self, hdrs, min_percent, exposures):
        mid = len(hdrs) // 2

        new_hdrs = []
        if self.nexps == 3:
            if exposures[mid] == 1:
                factor = np.random.uniform(0.1, 1)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # exposures[mid] == 4 or 8
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)
        else:
            if exposures[mid] == 1: # low exposure reference
                factor = np.random.uniform(0.1, 1)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # high exposure reference
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)

        for idx, hdr in enumerate(hdrs):
            new_hdr = (hdr / (anchor + 1e-8) * new_anchor).clip(0, 1)
            new_hdrs.append(new_hdr)

        ldrs = []
        for i in range(len(new_hdrs)):
            ldr = iutils.hdr_to_ldr(new_hdrs[i], exposures[i])
            ldrs.append(ldr)
        return new_hdrs[1:-1], ldrs, None

    def __len__(self):
        return len(self.patch_list) * self.repeat
