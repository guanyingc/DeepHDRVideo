"""
Dataloader for synthetic training dataset
We generate the synthetic training dataset using 21 videos from Cinematic Wide Gamut HDR-video and LiU HDRv Repository - Resources. To generate the synthetic training data, we cropped 7-frame HDR patches and save them as 16bit numpy arrays to hard disk before hand. 
[What this dataloader do] During training, We re-expose the HDR images into LDR images in an on-line manner
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

class syn_hdr_dataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(milestones=[5, 10, 20])
        return parser

    def __init__(self, args, split='train'):
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.nexps = args.nexps
        self._prepare_list(split)
        self._config_mode(split)

    def _prepare_list(self, split='train'):
        args = self.args
        list_name = split + '_list.txt'
        print("=> fetching img pairs in '{}'".format(args.hdr_data_dir1))
        patch_list1 = utils.read_list(os.path.join(args.hdr_data_dir1, list_name))
        full_patch_list1 = [os.path.join(args.hdr_data_dir1, 'Images', name) for name in patch_list1]

        print("=> fetching img pairs in '{}'".format(args.hdr_data_dir2))
        patch_list2 = utils.read_list(os.path.join(args.hdr_data_dir2, list_name))
        full_patch_list2 = [os.path.join(args.hdr_data_dir2, 'Images', name) for name in patch_list2]

        self.patch_list = patch_list1 + patch_list2
        self.full_patch_list = full_patch_list1 + full_patch_list2

    def _config_mode(self, split='train'):
        self.l_suffix = ''

        """ Use for debug only """
        if split == 'train' and self.args.split_train == 'high_only':
            self.l_suffix = '_high.txt'
        elif split == 'train' and self.args.split_train == 'low_only':
            self.l_suffix = '_low.txt'
        elif self.args.split_train not in ['', 'high_only', 'low_only']:
            raise Exception('Unknown split type')

        if self.nexps == 2:
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat
        else: # nexps == 3
            self.repeat = 2 if self.args.repeat <= 0 else self.args.repeat

    def __getitem__(self, index):
        hdr_paths = self._get_input_path(index)

        if self.nexps == 2:
            exposures = self._get_2exposures(index)
        elif self.nexps == 3:
            exposures = self._get_3exposures(index)
        else:
            raise Exception("Unknow exposures")

        hdrs = []
        for hdr_path in hdr_paths:
            hdrs.append(np.load(hdr_path).astype(np.float32)) # [0, 1]

        hdrs, imgs, anchor = self.re_expose_ldrs(hdrs, min_percent=97, exposures=exposures)
        num_hdrs = len(hdrs)
        imgs += hdrs

        imgs = self._augment_imgs(imgs)
        imgs, hdrs = imgs[:-len(hdrs)], imgs[-len(hdrs):]

        item = {}
        for i in range(1, self.nframes-1):
            item['hdr_%d' % i] = hdrs[i-1]

        for i in range(self.nframes):
            item['ldr_%d' % i] = imgs[i]

        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])
        item['expos'] = exposures
        return item

    def _augment_imgs(self, imgs):
        h, w, c = imgs[0].shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w

        if self.args.rescale and not (crop_h == h):
            max_h = h * self.args.sc_k
            max_w = w * self.args.sc_k
            sc_h = np.random.randint(crop_h, max_h) if self.args.rand_sc else self.args.scale_h
            sc_w = np.random.randint(crop_w, max_w) if self.args.rand_sc else self.args.scale_w
            imgs = hdr_transforms.rescale(imgs, [sc_h, sc_w])

        if self.args.flip_aug:
            imgs = hdr_transforms.random_flip_lrud(imgs)

        if self.args.rotate_aug:
            imgs = hdr_transforms.random_rotate(imgs, angle=10.0)

        if self.args.crop:
            imgs = hdr_transforms.random_crop(imgs, [crop_h, crop_w])

        if self.args.rotate90:
            imgs = hdr_transforms.random_rotate90(imgs)

        if self.args.cl_aug:
            color_permute = np.random.permutation(3)
            for i in range(len(imgs)):
                imgs[i] = imgs[i][:,:,color_permute]
        return imgs

    def _get_input_path(self, index):
        scene, patch = self.patch_list[index // self.repeat].split('/')
        img_dir = self.full_patch_list[index // self.repeat]

        data = np.genfromtxt(os.path.join(img_dir, patch + '.txt'), dtype='str', delimiter=' ')
        if self.args.static:
            """ for debug only, use static frames """
            img_idxs = np.ones(self.nframes, dtype=np.int)
        else:
            img_idxs = sorted(np.random.permutation(7)[:self.nframes])

        if self.args.inv_aug and np.random.random() > 0.5: # inverse time order
            img_idxs = img_idxs[::-1]
        data = data[img_idxs]

        hdr_paths = [os.path.join(img_dir, hdr + '.npy') for hdr in data[:, 0]]
        return hdr_paths

    def _get_2exposures(self, index):
        """ Generate exposure sequence for 2-exposure scene """
        cur_high = True if index % 2 == 0 else False # if middle frame is high exposure
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
        """ Generate exposure sequence for 2-exposure scene """
        cur_high = True if index % 2 == 0 else False
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
        """ min_percent is depreciated! can be ignore """
        # Re-expose HDR to LDRs
        mid = len(hdrs) // 2
        
        new_hdrs = []
        if self.nexps == 3:
            if exposures[mid] == 1:
                percent = np.random.uniform(99.9, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, anchor * 1.2)
            else:
                percent = np.random.uniform(97, 99.9)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)
        else:
            if exposures[mid] == 1: # low exposure
                percent = np.random.uniform(99.9, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, anchor * 1.2)
            else: # high exposure
                percent = np.random.uniform(97, 99.9)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)

        for idx, hdr in enumerate(hdrs):
            new_hdr = (hdr / (anchor + 1e-8) * new_anchor).clip(0, 1)
            new_hdrs.append(new_hdr)

        ldrs = []
        for i in range(len(new_hdrs)):
            ldr = iutils.hdr_to_ldr(new_hdrs[i], exposures[i])
            ldrs.append(ldr)
        return new_hdrs[1:-1], ldrs, anchor

    def __len__(self):
        return len(self.patch_list) * self.repeat
