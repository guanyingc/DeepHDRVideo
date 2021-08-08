"""
Dataloader for pre-aligned TOG13 dataset 
This file is not used in this repository, can be safely ignored.
"""
import os
import numpy as np
from imageio import imread
import cv2
import torch
import torch.utils.data as data

from datasets import hdr_transforms
from utils import utils
from utils import image_utils as iutils
np.random.seed(0)

class tog13_prealign_dataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, args, split='train'):
        self.root  = os.path.join(args.bm_dir)
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.expo_num = args.nexps
        self._prepare_list()

    def _prepare_list(self):
        suffix = '' if self.args.test_scene is None else self.args.test_scene
        if suffix == '':
            list_name = '2Exp_scenes.txt' if self.expo_num == 2 else '3Exp_scenes.txt'
            self.scene_list = utils.read_list(os.path.join(self.root, list_name))
        else:
            self.scene_list = utils.read_list(os.path.join(self.root, 'scenes_%s.txt' % suffix))
        #self.scene_list = utils.read_list(os.path.join(self.root, 'scenes%s.txt' % suffix))
        self.triple_list = self._collect_triple_list(self.scene_list, self.args.test_tog13_frames)
        print('[%s] totaling  %d triples' % (self.__class__.__name__, len(self.triple_list)))
    
    def _select_frames(self, t_list, max_frames=-1, skip_headtail=False):
        if max_frames > 0:
            idxs = self._sample_index(len(t_list), max_frames)
            t_list = [t_list[idx] for idx in idxs]
        if max_frames < 0 and skip_headtail:
            t_list = t_list[1:-1]
        return t_list

    def _sample_index(self, total_n, sample_n):
        if total_n == sample_n:
            return range(0, total_n)
        expo_num = self.expo_num
        pair = sample_n // expo_num
        prev_idx = np.linspace(0, total_n - expo_num, pair).reshape(-1, 1).astype(np.int)
        cur_idx = prev_idx + 1

        if expo_num == 2:
            idxs = np.concatenate([prev_idx, cur_idx], 1).reshape(-1)
        elif expo_num == 3:
            nxt_idx = cur_idx + 1
            idxs = np.concatenate([prev_idx, cur_idx, nxt_idx], 1).reshape(-1)
        else:
            raise Exception('Unknown expo_num: %d' % expo_num)
        # print(idxs)
        return idxs
    
    def __getitem__(self, index):
        index = index + self.args.start_idx
        img_paths, exposures, hdr_path = self._get_input_path(index)

        imgs = []
        for img_path in img_paths:
            if img_path[-3:] == 'tif':
                img = iutils.read_16bit_tif(img_path)
            else:
                img = imread(img_path) / 255.0
            imgs.append(img)
        mid = len(img_paths) // 2

        if os.path.exists(hdr_path):
            hdr = iutils.read_hdr(hdr_path).astype(np.float32)
        else:
            hdr = imgs[mid].copy()
        imgs.append(hdr)

        if self.args.test_resc:
            imgs = hdr_transforms.rescale(imgs, [self.args.test_h, self.args.test_w])
        hdr = imgs.pop()

        item = {}

        ldr_start, ldr_end, hdr_start, hdr_end = self.get_ldr_hdr_start_end(index)

        for i in range(ldr_start, ldr_end):
            item['ldr_%d' % i] = imgs[i]
        for i in range(hdr_start, hdr_end):
            item['hdr_%d' % i] = hdr # TODO:not the same

        origin_hw = (imgs[0].shape[0], imgs[0].shape[1])
        #if self.args.factor_of_k > 0:
        #    for k in item.keys():
        #        item[k] = hdr_transforms.imgsize_to_factor_of_k(item[k], self.args.factor_of_k)

        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])
        item['path'] = os.path.basename(hdr_path)
        item['scene'] = os.path.basename(os.path.dirname(os.path.dirname(hdr_path)))
        item['img_name'] = os.path.splitext(os.path.basename(img_paths[mid]))[0]
        item['expos'] = exposures
        item['hw'] = origin_hw
        item['reuse_cached_data'] = False
        return item

    def _get_input_path(self, index):
        img_dir = self.triple_list[index]
        list_name = os.path.basename(img_dir) + '.txt'
        img_expo_list = np.genfromtxt(os.path.join(img_dir, list_name), dtype='str')
        if len(img_expo_list) > self.nframes:
            start_i = (len(img_expo_list) - self.nframes) // 2
            img_expo_list = img_expo_list[start_i: start_i+self.nframes]
            
        img_paths = [os.path.join(img_dir, img) for img in img_expo_list[:, 0]]
        exposures = img_expo_list[:, 1].astype(np.float32)

        hdr_path = os.path.join(img_dir, os.path.basename(img_dir) + '.hdr')
        return img_paths, exposures, hdr_path

    def _collect_triple_list(self, scene_list, max_frames, skip_headtail=True): 
        triple_list = []
        for i in range(len(scene_list)):
            triple_dir = os.path.join(self.root, scene_list[i])
            triple_hard = False
            t_list = np.genfromtxt(os.path.join(triple_dir, 'pair_list.txt'), dtype='str')
            t_list = [os.path.join(triple_dir, triple) for triple in t_list]
            t_list = self._select_frames(t_list, max_frames, skip_headtail)

            print('[%s] [%d/%d] scene, %d triples' % 
                    (self.__class__.__name__, i + 1, len(self.scene_list), len(t_list)))
            triple_list += t_list
        return triple_list

    def get_ldr_hdr_start_end(self, index):
        ldr_start, ldr_end = 0, self.nframes
        if self.expo_num == 2:
            hdr_start, hdr_end = 1, self.nframes - 1
        elif self.expo_num == 3:
            hdr_start, hdr_end = 2, self.nframes - 2
        return ldr_start, ldr_end, hdr_start, hdr_end

