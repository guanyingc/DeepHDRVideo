"""
Dataloader for TOG13 dataset. The input images are not pre-aligned
Load similarity transformation matries for on-line alignment.
"""
import os
import numpy as np
from imageio import imread
import cv2
import torch
import torch.utils.data as data

from datasets import hdr_transforms
from utils import utils
import utils.image_utils as iutils
import scipy.io as sio
np.random.seed(0)

class tog13_online_align_dataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, args, split='train'):
        self.root  = os.path.join(args.bm_dir)
        self.split = split
        self.args  = args
        self.nframes = args.nframes
        self.expo_num = args.nexps if hasattr(args, 'nexps') else 2
        self._prepare_list()
        self.run_model = args.run_model if hasattr(args, 'run_model') else False
        #print(self.expos_list)

    def _prepare_list(self): 
        suffix = '' if self.args.test_scene is None else self.args.test_scene
        if suffix == '':
            list_name = '2Exp_scenes.txt' if self.expo_num == 2 else '3Exp_scenes.txt'
            self.scene_list = utils.read_list(os.path.join(self.root, list_name))
        else:
            self.scene_list = utils.read_list(os.path.join(self.root, 'scene_%s.txt' % suffix))

        self.crf = sio.loadmat(os.path.join(self.root, 'BaslerCRF.mat'))['BaslerCRF']
        skip_headtail = False 
        if self.nframes == 3 or (self.nframes == 5 and self.expo_num == 3):
            skip_headtail = True
    
        self._collect_triple_list(self.scene_list, self.args.test_tog13_frames, skip_headtail=skip_headtail)
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

            img_dir = os.path.join(self.root, scene_list[i])

            img_list, hdr_list = self._load_img_hdr_list(img_dir)
            e_list = self._load_exposure_list(os.path.join(img_dir, 'Exposures.txt'), img_num=len(img_list))
            idx_list = list(range(self.total_img_num, self.total_img_num + len(img_list))) # each image has a unique index
            self.total_img_num += len(img_list)

            img_list, hdr_list, e_list, idx_list = self._lists_to_paired_lists([img_list, hdr_list, e_list, idx_list])

            if self.expo_num == 2:
                hdr_list = [h_list[1:-1] for h_list in hdr_list] # N * (nframes - 2)
            elif self.expo_num == 3:
                hdr_list = [h_list[2:-2] for h_list in hdr_list]

            img_list, e_list, hdr_list, idx_list = self._select_frames([img_list, e_list, hdr_list, idx_list], max_frames, skip_headtail)
            
            print('[%s] [%d/%d] scene, %d triples' % 
                    (self.__class__.__name__, i + 1, len(self.scene_list), len(img_list)))
            self.expos_list += e_list
            self.triple_list += img_list
            self.hdrs_list += hdr_list
            self.idxs_list += idx_list
        # print(self.idxs_list)

    def _load_img_hdr_list(self, img_dir):
        if os.path.exists(os.path.join(img_dir, 'img_hdr_list.txt')):
            img_hdr_list = np.genfromtxt(os.path.join(img_dir, 'img_hdr_list.txt'), dtype='str')
            img_list = img_hdr_list[:, 0]
            hdr_list = img_hdr_list[:, 1]
        else:
            img_list = np.genfromtxt(os.path.join(img_dir, 'img_list.txt'), dtype='str')
            hdr_list = ['None'] * len(img_list)
        img_list =[os.path.join(img_dir, img_path) for img_path in img_list]
        hdr_list =[os.path.join(img_dir, hdr_path) for hdr_path in hdr_list]
        return img_list, hdr_list

    def _load_exposure_list(self, expos_path, img_num):
        expos = np.genfromtxt(expos_path, dtype='float')
        expos = np.power(2, expos - expos.min()).astype(np.float32)
        expo_list = np.tile(expos, int(img_num / len(expos) + 1))[:img_num]
        return expo_list

    def _lists_to_paired_lists(self, lists):
        paired_lists = []

        for l in lists:
            paired_list = []
            paired_list.append(l[: len(l) - self.nframes + 1])
            for j in range(1, self.nframes):
                start_idx, end_idx = j, len(l) - self.nframes + 1 + j
                paired_list.append(l[start_idx: end_idx])

            paired_lists.append(np.stack(paired_list, 1).tolist()) # Nxframes
        return paired_lists

    def _select_frames(self, lists, max_frames=-1, skip_headtail=False):
        if max_frames > 0:
            max_frames = min(max_frames, len(lists[0]))
            idxs = self._sample_index(len(lists[0]), max_frames)
            for i in range(len(lists)):
                lists[i] = [lists[i][j] for j in idxs]

        if max_frames < 0 and skip_headtail:
            for i in range(len(lists)):
                lists[i] = lists[i][1:-1]
        return lists

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
        img_paths, exposures, hdr_paths = self._get_input_path(index)

        item = {}
        ldr_start, ldr_end, hdr_start, hdr_end = self.get_ldr_hdr_start_end(index)
        
        for i in range(ldr_start, ldr_end):
            img = iutils.apply_gamma(iutils.read_16bit_tif(img_paths[i], crf=self.crf), gamma=2.2)
            item['ldr_%d' % i] = img

        for i in range(hdr_start, hdr_end):
            if self.expo_num == 2:
                hdr_path = hdr_paths[i-1]
            elif self.expo_num == 3:
                hdr_path = hdr_paths[i-2]

            if os.path.exists(hdr_path):
                hdr = iutils.read_hdr(hdr_path)
                if hdr.max() > 1:
                    hdr = hdr / hdr.max()
            else:
                # No GT hdr, use linearized LDR as GT
                if 'ldr_%d'%i in item:
                    hdr = iutils.ldr_to_hdr(item['ldr_%d'%i], exposures[i])
                else: # run_model mode
                    hdr = iutils.ldr_to_hdr(item['ldr_%d'%ldr_start], exposures[ldr_start])
            item['hdr_%d' % i] = hdr
        
        origin_hw = (img.shape[0], img.shape[1])
        item = self.post_process(item, img_paths)

        if self.args.align:
            print('Loading affine transformation matrices for online-alignment')
            for i in range(ldr_start, ldr_end):
                item['match_%d' % i] = self.load_affine_matrices(img_paths[i], img.shape[0], img.shape[1])

        item['expos'] = exposures
        item['hw'] = origin_hw
        item['reuse_cached_data'] = True if ldr_end - ldr_start == 1 else False
        return item

    def _get_input_path(self, index, get_img_idx=False):
        img_paths, hdr_paths = self.triple_list[index], self.hdrs_list[index]
        exposures = np.array(self.expos_list[index]).astype(np.float32)

        if get_img_idx:
            img_idxs = self.idxs_list[index]
            return img_paths, exposures, hdr_paths, img_idxs
        else:
            return img_paths, exposures, hdr_paths

    def post_process(self, item, img_paths):
        #if self.args.factor_of_k > 0:
        #    for k in item.keys():
        #        item[k] = hdr_transforms.imgsize_to_factor_of_k(item[k], self.args.factor_of_k)

        for k in item.keys(): 
            item[k] = hdr_transforms.array_to_tensor(item[k])

        mid_img_path = img_paths[len(img_paths)//2]
        item['scene'] = os.path.basename(os.path.dirname(mid_img_path))
        item['img_name'] = os.path.splitext(os.path.basename(mid_img_path))[0]
        return item

    def __len__(self):
        return len(self.triple_list)

    def load_affine_matrices(self, img_path, h, w):
        dir_name, img_name = os.path.dirname(img_path), os.path.basename(img_path)
        cv2_match = np.genfromtxt(os.path.join(dir_name, 'Affine_Trans_Matrices', img_name[:-4]+'_match.txt'), dtype=np.float32)
        # For two exposure: cv2_match [2, 6], row 1->2: cur->prev, cur->next
        # For three exposure: cv2_match [4, 6], row1->4: cur->prev2, cur->prev, cur->next, cur->next2

        n_matches =cv2_match.shape[0]
        if self.args.nexps == 2:
            assert (n_matches == 2)
        elif self.args.nexps == 3:
            assert (n_matches == 4)

        cv2_match = cv2_match.reshape(n_matches, 2, 3)
        theta = np.zeros((n_matches, 2, 3)).astype(np.float32) # Theta for affine transformation in pytorch
        for mi in range(n_matches):
            theta[mi] = hdr_transforms.cvt_MToTheta(cv2_match[mi], w, h)
        return theta

    def get_ldr_hdr_start_end(self, index):
        if self.run_model and self.args.cached_data and index not in self.scene_st_idxs:
            ldr_start = self.nframes - 1
            ldr_end = ldr_start + 1
            if self.expo_num == 2: # 3-2=1, 5-2=3
                hdr_start = self.nframes - 2
            elif self.expo_num == 3: # 5-3=2, 7-3=4
                hdr_start = self.nframes - 3
            hdr_end = hdr_start + 1
        else:
            ldr_start, ldr_end = 0, self.nframes
            if self.expo_num == 2:
                hdr_start, hdr_end = 1, self.nframes - 1
            elif self.expo_num == 3:
                hdr_start, hdr_end = 2, self.nframes - 2
        return ldr_start, ldr_end, hdr_start, hdr_end

