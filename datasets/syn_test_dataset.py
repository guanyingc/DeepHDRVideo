"""
Dataloader for synthetic test dataset
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

class syn_test_dataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, args, split='test'):
        self.root  = os.path.join(args.bm_dir)
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.nexps = args.nexps
        self._prepare_list()
        self.run_model = args.run_model if hasattr(args, 'run_model') else False

    def _prepare_list(self):
        suffix = '' if self.args.test_scene is None else self.args.test_scene
        if suffix == '':
            list_name = 'scenes_2expo.txt' if self.nexps == 2 else 'scenes_3expo.txt'
            self.scene_list = utils.read_list(os.path.join(self.root, list_name))
        else:
            self.scene_list = utils.read_list(os.path.join(self.root, 'scene_%s.txt' % suffix))

        self._collect_triple_list(self.scene_list, self.args.test_syn_frames, skip_headtail=False)
        print('[%s] totaling  %d triples' % (self.__class__.__name__, len(self.triple_list)))

    def _collect_triple_list(self, scene_list, max_frames, skip_headtail=True): 
        self.expos_list = []
        self.triple_list = []
        self.scene_st_idxs = [] # triple index of a new scene
        self.hdrs_list = []
        self.idxs_list = []
        
        self.total_img_num = 0
        for i in range(len(scene_list)):
            self.scene_st_idxs.append(len(self.triple_list))

            img_dir = os.path.join(self.root, 'Images', scene_list[i])

            img_list, hdr_list = self._load_img_hdr_list(img_dir)

            e_list = self._load_exposure_list(os.path.join(img_dir, 'Exposures.txt'), img_num=len(img_list))
            idx_list = list(range(self.total_img_num, self.total_img_num + len(img_list))) # each image has a unique index
            self.total_img_num += len(img_list)

            img_list, hdr_list, e_list, idx_list = self._lists_to_paired_lists([img_list, hdr_list, e_list, idx_list])

            if self.nexps == 2:
                hdr_list = [h_list[1:-1] for h_list in hdr_list] # N * (nframes - 2)
            elif self.nexps == 3:
                hdr_list = [h_list[2:-2] for h_list in hdr_list]

            #img_list, e_list, hdr_list, idx_list = self._select_frames([img_list, e_list, hdr_list, idx_list], max_frames, skip_headtail)
            
            print('[%s] [%d/%d] scene, %d triples' % 
                    (self.__class__.__name__, i + 1, len(self.scene_list), len(img_list)))
            self.expos_list += e_list
            self.triple_list += img_list
            self.hdrs_list += hdr_list
            self.idxs_list += idx_list
        # print(self.idxs_list)

    def _load_img_hdr_list(self, img_dir):
        scene_list = np.genfromtxt(os.path.join(img_dir, 'img_list.txt'), dtype='str')
        img_list = ['%s.tif' % name for name in scene_list]
        hdr_list = ['%s.hdr' % name for name in scene_list]
        img_list =[os.path.join(img_dir, img_path) for img_path in img_list]
        hdr_list =[os.path.join(img_dir, hdr_path) for hdr_path in hdr_list]
        return img_list, hdr_list

    def _load_exposure_list(self, expos_path, img_num):
        expos = np.genfromtxt(expos_path, dtype='float')
        expos = np.power(2, expos - expos.min()).astype(np.float32)
        expo_list = np.tile(expos, int(img_num / len(expos) + 1))[:img_num]
        return expo_list

    def __getitem__(self, index):
        index = index + self.args.start_idx
        img_paths, exposures, hdr_paths = self._get_input_path(index)
        #print(img_paths, exposures, hdr_paths)

        item = {}
        ldr_start, ldr_end, hdr_start, hdr_end = self.get_ldr_hdr_start_end(index)
        
        for i in range(ldr_start, ldr_end):
            img = iutils.read_16bit_tif(img_paths[i])
            item['ldr_%d' % i] = img

        for i in range(hdr_start, hdr_end):
            if self.nexps == 2:
                hdr_path = hdr_paths[i-1]
            elif self.nexps == 3:
                hdr_path = hdr_paths[i-2]

            if os.path.exists(hdr_path):
                hdr = iutils.read_hdr(hdr_path)
            else:
                # No GT hdr, use linearized LDR as GT
                if 'ldr_%d'%i in item:
                    hdr = iutils.ldr_to_hdr(item['ldr_%d'%i], exposures[i])
                else: # run_model mode
                   hdr = iutils.ldr_to_hdr(item['ldr_%d'%ldr_start], exposures[ldr_start])
            item['hdr_%d' % i] = hdr

        origin_hw = (img.shape[0], img.shape[1])
        item = self.post_process(item, img_paths)
        item['expos'] = exposures
        item['hw'] = origin_hw
        item['reuse_cached_data'] = True if ldr_end - ldr_start == 1 else False
        return item

    def _get_input_path(self, index, get_img_idx=False):
        img_paths, hdr_paths = self.triple_list[index], self.hdrs_list[index]
        exposures = np.array(self.expos_list[index]).astype(np.float32)

        return img_paths, exposures, hdr_paths

    def post_process(self, item, img_paths):
        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])

        mid_img_path = img_paths[len(img_paths)//2]
        item['scene'] = os.path.basename(os.path.dirname(mid_img_path))
        item['img_name'] = os.path.splitext(os.path.basename(mid_img_path))[0]
        return item

    def get_ldr_hdr_start_end(self, index):
        if self.run_model and self.args.cached_data and index not in self.scene_st_idxs:
            ldr_start = self.nframes - 1
            ldr_end = ldr_start + 1
            if self.nexps == 2: # 3-2=1, 5-2=3
                hdr_start = self.nframes - 2
            elif self.nexps == 3: # 5-3=2, 7-3=4
                hdr_start = self.nframes - 3
            hdr_end = hdr_start + 1
        else:
            ldr_start, ldr_end = 0, self.nframes
            if self.nexps == 2:
                hdr_start, hdr_end = 1, self.nframes - 1
            elif self.nexps == 3:
                hdr_start, hdr_end = 2, self.nframes - 2
        return ldr_start, ldr_end, hdr_start, hdr_end

    def _lists_to_paired_lists(self, lists):
        paired_lists = []

        for l in lists:
            if (self.nexps == 2 and self.nframes == 3) or (self.nexps == 3 and self.nframes == 5):
                l = l[1:-1]
            paired_list = []
            paired_list.append(l[: len(l) - self.nframes + 1])
            for j in range(1, self.nframes):
                start_idx, end_idx = j, len(l) - self.nframes + 1 + j
                paired_list.append(l[start_idx: end_idx])

            paired_lists.append(np.stack(paired_list, 1).tolist()) # Nxframes
        return paired_lists

    def __len__(self):
        return len(self.triple_list)
