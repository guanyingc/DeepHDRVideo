"""
Single-stage model (2-exposure)
This model contains a flownet for optical flow alignment and a weight net for HDR fusion
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from utils import eval_utils as eutils
from models import model_utils as mutils
from models import network_utils as nutils
from collections import OrderedDict
from models.hdr2E_model import hdr2E_model
np.random.seed(0)

class hdr2E_flow_model(hdr2E_model):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--fnet_name', default='spynet_triple') # for 2 exposures
        parser.add_argument('--fnet_checkp', default='')
        parser.add_argument('--mnet_name', default='weight_net')
        parser.add_argument('--mnet_checkp', default='')
        parser.add_argument('--mnet_afunc', default='LReLU')
        parser.add_argument('--test_root', default='mnet_checkp', help='path for saving test results')
        parser.add_argument('--nframes', type=int, default=3, help='number of input frames')
        parser.add_argument('--nexps', type=int, default=2, help='number of exposures')

        parser.add_argument('--cmid', default=256, type=int, help='channel number of the bottle layer in weight net')
        parser.add_argument('--up_fnet', default=True, action='store_false', help='if update flownet')
        parser.add_argument('--fnet_init', default='xavier')
        parser.add_argument('--tone_low', default=False, action='store_true')
        parser.add_argument('--tone_ref', default=True, action='store_false')

        parser.set_defaults(factor_of_k=32) # resize the image size to be factor of 32

        str_keys = ['fnet_name', 'mnet_name', 'mnet_afunc']
        val_keys = ['cmid']
        bool_keys = []
        bool_val_dicts = {}
        bool_to_val_dicts = {}
        return parser, str_keys, val_keys, bool_keys, bool_val_dicts, bool_to_val_dicts

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.net_names = ['fnet', 'mnet']

        c_in_fnet, c_out_fnet = 9, 4
        other_opt = {}
        self.fnet = self.import_network(args.fnet_name, backup_module='flow_networks')(
                c_in=c_in_fnet, c_out=c_out_fnet, requires_grad=self.opt['up_fnet'], other=other_opt)
        self.fnet = mutils.init_net(self.fnet, init_type=opt['fnet_init'], gpu_ids=args.gpu_ids)
        
        c_in_mnet, c_out_mnet = self.get_io_ch_nums(opt)

        self.mnet = self.import_network(args.mnet_name)(c_in_mnet, c_out_mnet, c_mid=opt['cmid'], 
                use_bn=opt['use_bn'], afunc=opt['mnet_afunc'])
        self.mnet = mutils.init_net(self.mnet, init_type=opt['init_type'], gpu_ids=args.gpu_ids)

        if self.is_train: # Criterion
            self.config_optimizers(opt, log)
        self.config_criterions(opt, log)

        self.load_checkpoint(log)
        self.backward_grid = {}
        torch.manual_seed(0)

    def get_io_ch_nums(self, opt):
        c_in = 9
        if opt['m_in_nb']:
            c_in += 2 * 3 # adding two neighboring frames
        c_out = c_in

        if opt['in_ldr']:
            c_in *= 2 # adding ldr images

        return c_in, c_out

    def config_optimizers(self, opt, log):
        params = list(self.mnet.parameters())
        if self.opt['up_fnet']:
            params += list(self.fnet.parameters())
        self.optimizer = torch.optim.Adam(params, lr=opt['init_lr'], 
                            betas=(opt['beta_1'], opt['beta_2']))
        self.optimizers.append(self.optimizer)
        self.setup_lr_scheduler() # defined in base model

    def forward(self, split='train'):
        self.split = split
        self.prepare_inputs(self.data)
        fnet_in = self.prepare_fnet_input()
        self.fpred = self.fnet(fnet_in)

        mnet_in, merge_hdrs, warped_imgs = self.prepare_mnet_inputs(self.opt, self.data, self.fpred, idxs=[0, 1, 2])

        self.pred = self.mnet(mnet_in, merge_hdrs)

        if self.opt['mask_o']:
            mask = self.data['gt_ref_ws'][self.hdr_mid]
            self.pred['hdr'] = self.data['l2hdrs'][self.ldr_mid] * mask + self.pred['hdr'] * (1 - mask)

        self.pred['log_hdr'] = eutils.pt_mulog_transform(self.pred['hdr'])

        self.pred.update(warped_imgs)
        
        if not self.is_train:
            self.cached_data = self.data
        self.loss_terms = None
        return self.pred

    def prepare_mnet_inputs(self, opt, data, fpred, idxs, in_cur_hdr=False):
        """
        Require ldrs, l2hdrs
        """
        pi, ci, ni = idxs
        backward_grid = self.backward_grid
        shape = data['ldrs'][ci].shape
        if str(shape) not in backward_grid:
            backward_grid[str(shape)] = nutils.generate_grid(data['ldrs'][ci])
        
        use_clean_hdr_in = self.split in ['train'] and self.opt['clean_in']

        if use_clean_hdr_in:
            p_warp = nutils.backward_warp(data['clean_ldrs'][pi], fpred['flow1'], backward_grid[str(shape)])
            n_warp = nutils.backward_warp(data['clean_ldrs'][ni], fpred['flow2'], backward_grid[str(shape)])
        else:
            p_warp = nutils.backward_warp(data['ldrs'][pi], fpred['flow1'], backward_grid[str(shape)])
            n_warp = nutils.backward_warp(data['ldrs'][ni], fpred['flow2'], backward_grid[str(shape)])
        p_warp_hdr = mutils.pt_ldr_to_hdr(p_warp, data['expos'][pi])
        n_warp_hdr = mutils.pt_ldr_to_hdr(n_warp, data['expos'][ni])

        ldrs = [data['ldrs'][ci], p_warp, n_warp]
        hdrs = [data['l2hdrs'][ci], p_warp_hdr, n_warp_hdr]
    
        if use_clean_hdr_in:
            merge_hdrs = [data['clean_l2hdrs'][ci], p_warp_hdr, n_warp_hdr]
        elif self.split in ['train', 'val'] and (self.opt['tone_ref'] or in_cur_hdr):
            merge_hdrs = [data['in_cur_hdr'], p_warp_hdr, n_warp_hdr]
        else:
            merge_hdrs = [data['l2hdrs'][ci], p_warp_hdr, n_warp_hdr]

        if opt['m_in_nb']:
            ldrs += [data['ldrs'][pi], data['ldrs'][ni]]
            hdrs += [data['l2hdrs'][pi], data['l2hdrs'][ni]]

            if use_clean_hdr_in:
                merge_hdrs += [data['clean_l2hdrs'][pi], data['clean_l2hdrs'][ni]]
            else:
                merge_hdrs += [data['l2hdrs'][pi], data['l2hdrs'][ni]]

        net_in = hdrs
        if opt['in_ldr']: 
            net_in += ldrs

        net_in = torch.cat(net_in, 1)
        warped_imgs = {'p_warp': p_warp, 'n_warp': n_warp}
        return net_in, merge_hdrs, warped_imgs

    def prepare_fnet_input(self):
        data = self.data
        fnet_in = [self.data['ldrs'][0], self.data['ldr_adjs'][1], self.data['ldrs'][2]]
        return fnet_in

    def optimize_weights(self):
        self.loss_terms = OrderedDict()
        data, pred = self.data, self.pred

        if self.opt['mask_o']:
            roi = 1 - self.data['gt_ref_ws'][self.hdr_mid]
            hdr_loss = self.hdr_crit(self.pred['log_hdr'] * roi, self.data['log_hdrs'][self.hdr_mid] * roi) / (roi.mean() + 1e-8)
            self.loss_terms['mhdr_loss'] = hdr_loss.item()
        else:
            hdr_loss = self.hdr_crit(self.pred['log_hdr'], self.data['log_hdrs'][self.hdr_mid])
            self.loss_terms['hdr_loss'] = hdr_loss.item()
        self.loss = self.opt['hdr_w'] * hdr_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
   
    def prepare_records(self):
        data, pred = self.data, self.pred
        records, iter_res = hdr2E_model.prepare_records(self)
        self.prepare_flow_info(records)

        return records, iter_res

    def prepare_flow_info(self, records, flow=None, key=''):
        if flow is None:
            flow = self.fpred['flow1'].detach()
        f_min, f_mean, f_max = flow.min(), flow.abs().mean(), flow.max()
        records['%sf_min_range' % key]  = f_min.item()
        records['%sf_mean_range' % key] = f_mean.item()
        records['%sf_max_range' % key]  = f_max.item()

    def prepare_visual(self):
        data, pred = self.data, self.pred
        fpred = self.fpred
        visuals = []
        
        visuals += [data['log_hdrs'][self.hdr_mid], pred['log_hdr']]
        diff = eutils.pt_cal_diff_map(self.pred['log_hdr'].detach(), data['log_hdrs'][self.hdr_mid])
        visuals.append(eutils.pt_colormap(diff))

        if self.opt['mask_o']:
            visuals += [self.data['gt_ref_ws'][self.hdr_mid]]

        for i, ldr in enumerate(data['ldrs']):
            visuals += [ldr]

        for i, ldr_adj in enumerate(data['ldr_adjs']):
            visuals += [ldr_adj]

        visuals += [pred['p_warp'], pred['n_warp']]
        visuals.append(eutils.pt_blend_images(data['ldrs']))

        flow1_color = eutils.pt_flow_to_color(fpred['flow1'].detach())
        flow2_color = eutils.pt_flow_to_color(fpred['flow2'].detach())
        visuals += [flow1_color, flow2_color]

        if 'weights' in self.pred:
            visuals += self.pred['weights']

        if self.split not in ['train', 'val'] and self.opt['origin_hw']:
            new_visuals = eutils.crop_list_of_tensors(visuals, data['hw'])
            return new_visuals
        return visuals
