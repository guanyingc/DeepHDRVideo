"""
Single-stage model (3-exposure)
This model contains a flownet for optical flow alignment and a weight net for HDR fusion
"""
import os
import torch
import numpy as np
from .base_model import BaseModel
from utils import eval_utils as eutils
from models import model_utils as mutils
from models import network_utils as nutils
from models import noise_utils as noutils
from collections import OrderedDict
from models.hdr2E_model import hdr2E_model
np.random.seed(0)

class hdr3E_flow_model(hdr2E_model):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--fnet_name', default='spynet_2triple') # for 3 exposures
        parser.add_argument('--fnet_checkp', default='')
        parser.add_argument('--mnet_name', default='weight_net')
        parser.add_argument('--mnet_checkp', default='')
        parser.add_argument('--mnet_afunc', default='LReLU')
        parser.add_argument('--test_root', default='mnet_checkp')
        parser.add_argument('--nframes', type=int, default=5)
        parser.add_argument('--nexps', type=int, default=3)

        parser.add_argument('--cmid', default=256, type=int)
        parser.add_argument('--up_fnet', default=True, action='store_false')
        parser.add_argument('--fnet_init', default='xavier')
        parser.add_argument('--tone_low', default=False, action='store_true')
        parser.add_argument('--tone_ref', default=True, action='store_false')

        parser.set_defaults(factor_of_k=32, init_lr=0.0001, mask_o=False, use_bn=False)

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
        c_in_fnet, c_out_fnet = 9 * 2, 8
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

    def get_io_ch_nums(self, opt):
        c_in = 15
        if opt['m_in_nb']:
            c_in += 4 * 3
        c_out = c_in

        if opt['in_ldr']:
            c_in *= 2
        return c_in, c_out

    def config_optimizers(self, opt, log):
        params = list(self.mnet.parameters())
        if self.opt['up_fnet']:
            params += list(self.fnet.parameters())
        self.optimizer = torch.optim.Adam(params, lr=opt['init_lr'], 
                            betas=(opt['beta_1'], opt['beta_2']))
        self.optimizers.append(self.optimizer)
        self.setup_lr_scheduler() # defined in base model

    def prepare_inputs(self, data):
        self.nframes = self.opt['nframes']
        self.nexps = self.opt['nexps']
        self.ldr_mid, self.hdr_mid = self.nframes // 2, (self.nframes - 4) // 2

        if not self.is_train and self.opt['cached_data'] and data['reuse_cached_data']: # hasattr(self, 'cached_data')
            self.reuse_cached_data(data)
            return

        # Exposure information
        expos = data['expos'].view(-1, self.nframes, 1, 1).split(1, 1)
        # HDR images
        hdrs, log_hdrs = [], []

        for i in range(2, self.nframes-2):
            hdrs.append(data['hdr_%d' % i])
            log_hdrs.append(eutils.pt_mulog_transform(data['hdr_%d' % i], self.mu))
       
        # LDR images
        ldrs, l2hdrs = [], []
        for i in range(self.nframes):
            ldrs.append(data['ldr_%d'%i])

         # Online global alignmnet
        if self.opt['align']:
            matches = []
            for i in range(self.nframes):
                matches.append(data['match_%d' % i])
            data['matches'] = matches

        gt_ref_ws = []
        for i in range(2, len(ldrs)-2):
            if self.nexps == 3:
                h_idx, m_idx, l_idx = self.get_expo_level_idxs(expos, frame_idx=i)
                gt_ref_ws.append(1.0 - self.get_out_mask_method()(ldrs[i], h_idx, m_idx=m_idx, l_idx=l_idx, h_thr=self.opt['o_hthr'], l_thr=self.opt['o_lthr']))


        tone_aug = self.opt['tone_low'] + self.opt['tone_ref']
        assert tone_aug <= 1

        if self.split in ['train', 'val'] and tone_aug == 1:
            self.perturb_low_expo_imgs(ldrs, expos)

            if self.opt['tone_ref']:
                data['in_cur_hdr'] = mutils.pt_ldr_to_hdr(ldrs[self.ldr_mid], expos[self.ldr_mid])
                ldrs[self.ldr_mid] = noutils.pt_tone_ref_tone_augment(ldrs[self.ldr_mid], d=0.7)

        for i in range(self.nframes):
            l2hdrs.append(mutils.pt_ldr_to_hdr(ldrs[i], expos[i]))

        assert(len(expos) == len(ldrs))

        c2p2_adjs, c2p1_adjs = [], []
        for i in range(2, self.nframes - 2):
            c2p2_adjs.append(mutils.pt_ldr_to_ldr(ldrs[i], expos[i], expos[i-2]))
            c2p1_adjs.append(mutils.pt_ldr_to_ldr(ldrs[i], expos[i], expos[i-1]))

        data.update({'hdrs': hdrs, 'log_hdrs': log_hdrs, 
            'ldrs': ldrs, 'l2hdrs': l2hdrs, 'c2p2_adjs': c2p2_adjs, 'c2p1_adjs': c2p1_adjs,
            'expos': expos,  'gt_ref_ws': gt_ref_ws}) # 'expms': expms,

    def perturb_low_expo_imgs(self, ldrs, expos):
        for i in range(self.nexps):
            tone_d = None
            cur_l_idx = (expos[i] == 1).view(-1)
            cur_m_idx = (expos[i] == 4).view(-1)
            params = {}
            if cur_l_idx.sum() > 0:
                for j in range(i, self.nframes, self.nexps): # e.g., [0,2], [0,3]
                    if self.opt['tone_ref']:
                        ldrs[j][cur_l_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-3, stdv2=1e-3, scale=False)
                    elif self.opt['tone_low']:
                        ldrs[j][cur_l_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-4, stdv2=1e-3, scale=False)
                    else:
                        raise Exception('Unknown tone low mode')

            if cur_m_idx.sum() > 0:
                for j in range(i, self.nframes, self.nexps): # e.g., [0,2], [0,3]
                    if self.opt['tone_low']:
                        ldrs[j][cur_l_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-4, stdv2=1e-3, scale=False)
                    elif self.opt['tone_ref']:
                        params['noise1'] = True; params['tone'] = False
                        ldrs[j][cur_m_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_m_idx], stdv1=1e-3, stdv2=1e-3, scale=False)
                    else:
                        raise Exception('Unknown tone low mode')

    def reuse_cached_data(self, data):
        print('Reused cached data')
        ldr_idx, hdr_idx = self.nframes - 1, self.nframes - 3
        #print(data.keys())
        
        reused_data_key = ['hdrs', 'log_hdrs', 'ldrs', 'l2hdrs', 'c2p2_adjs', 'c2p1_adjs', 'gt_ref_ws'] #, 'expms'
        if self.opt['align']:
            reused_data_key.append('matches')

        for key in reused_data_key:
            data[key] = self.cached_data[key]
            data[key].pop(0)

        hdr = data['hdr_%d' % hdr_idx]
        ldr = data['ldr_%d' % ldr_idx]
        data['hdrs'].append(hdr)
        data['log_hdrs'].append(eutils.pt_mulog_transform(data['hdr_%d' % hdr_idx], self.mu))
        data['ldrs'].append(ldr)
        
        if self.opt['align']:
            data['matches'].append(data['match_%d'%ldr_idx])

        expos = data['expos'].view(-1, self.nframes, 1, 1).split(1, 1)
        data['expos'] = expos

        data['l2hdrs'].append(mutils.pt_ldr_to_hdr(ldr, expos[ldr_idx]))

        h_idx, m_idx, l_idx = self.get_expo_level_idxs(expos, frame_idx=ldr_idx)
        
        h_idx, m_idx, l_idx = self.get_expo_level_idxs(expos, frame_idx=hdr_idx)
        data['gt_ref_ws'].append(1.0 - self.get_out_mask_method()(data['ldrs'][hdr_idx], h_idx, m_idx=m_idx, l_idx=l_idx, h_thr=self.opt['o_hthr'], l_thr=self.opt['o_lthr']))

        data['c2p2_adjs'].append(mutils.pt_ldr_to_ldr(data['ldrs'][hdr_idx], expos[hdr_idx], expos[hdr_idx-2]))
        data['c2p1_adjs'].append(mutils.pt_ldr_to_ldr(data['ldrs'][hdr_idx], expos[hdr_idx], expos[hdr_idx-1]))

    def get_expo_level_idxs(self, expos, frame_idx):
        i = frame_idx
        if i == 0:
            h_idx = (expos[i] > expos[i+1]).view(-1) & (expos[i] > expos[i+2]).view(-1)
            l_idx = (expos[i] < expos[i+1]).view(-1) & (expos[i] < expos[i+2]).view(-1)
        elif i == len(expos)-1:
            h_idx = (expos[i] > expos[i-1]).view(-1) & (expos[i] > expos[i-2]).view(-1)
            l_idx = (expos[i] < expos[i-1]).view(-1) & (expos[i] < expos[i-2]).view(-1)
        else: # 0 < i < len(expos)-1
            h_idx = (expos[i] > expos[i-1]).view(-1) & (expos[i] > expos[i+1]).view(-1)
            l_idx = (expos[i] < expos[i-1]).view(-1) & (expos[i] < expos[i+1]).view(-1)
        m_idx = 1 - h_idx - l_idx
        return h_idx, m_idx, l_idx

    def get_out_mask_method(self):
        if self.opt['soft_mo']:
            get_out_mask_method = mutils.pt_get_out_blend_mask
        else:
            get_out_mask_method = mutils.pt_ldr_to_3exps_1c_mask
        return get_out_mask_method

    def forward(self, split='train'):
        self.split = split
        self.prepare_inputs(self.data) 
        fnet_in = self.prepare_fnet_input()
        self.fpred = self.fnet(fnet_in)

        mnet_in, merge_hdrs, warped_imgs = self.prepare_mnet_inputs(self.opt, self.data, self.fpred, idxs=[0, 1, 2, 3, 4])

        self.pred = self.mnet(mnet_in, merge_hdrs)

        if self.opt['mask_o']:
            mask = self.data['gt_ref_ws'][self.hdr_mid]
            self.pred['hdr'] = self.data['l2hdrs'][self.ldr_mid] * mask + self.pred['hdr'] * (1 - mask)
        self.pred['log_hdr'] = eutils.pt_mulog_transform(self.pred['hdr'], self.mu)

        self.pred.update(warped_imgs)

        if not self.is_train:
            self.cached_data = self.data
        self.loss_terms = None
        return self.pred

    def prepare_mnet_inputs(self, opt, data, fpred, idxs):
        p2, p1, ci, n1, n2 = idxs
        backward_grid = self.backward_grid
        shape = data['ldrs'][ci].shape
        if str(shape) not in backward_grid:
            backward_grid[str(shape)] = nutils.generate_grid(data['ldrs'][ci])

        p2_warp = nutils.backward_warp(data['ldrs'][p2], fpred['flow1'], backward_grid[str(shape)])
        n1_warp = nutils.backward_warp(data['ldrs'][n1], fpred['flow2'], backward_grid[str(shape)])
        p2_warp_hdr = mutils.pt_ldr_to_hdr(p2_warp, data['expos'][p2])
        n1_warp_hdr = mutils.pt_ldr_to_hdr(n1_warp, data['expos'][n1])

        p1_warp = nutils.backward_warp(data['ldrs'][p1], fpred['flow3'], backward_grid[str(shape)])
        n2_warp = nutils.backward_warp(data['ldrs'][n2], fpred['flow4'], backward_grid[str(shape)])
        p1_warp_hdr = mutils.pt_ldr_to_hdr(p1_warp, data['expos'][p1])
        n2_warp_hdr = mutils.pt_ldr_to_hdr(n2_warp, data['expos'][n2])

        hdrs = [data['l2hdrs'][ci], p2_warp_hdr, n1_warp_hdr, p1_warp_hdr, n2_warp_hdr]
        
        # Neighbouring frames
        hdrs += [data['l2hdrs'][p2], data['l2hdrs'][n1], data['l2hdrs'][p1], data['l2hdrs'][n2]]

        if self.split in ['train', 'val'] and self.opt['tone_ref']:
            merge_hdrs = [data['in_cur_hdr'], p2_warp_hdr, n1_warp_hdr, p1_warp_hdr, n2_warp_hdr]
            merge_hdrs += [data['l2hdrs'][p2], data['l2hdrs'][n1], data['l2hdrs'][p1], data['l2hdrs'][n2]]
        else:
            merge_hdrs = hdrs        

        inputs = []
        if opt['in_ldr']:
            ldrs = [data['ldrs'][ci], p2_warp, n1_warp, p1_warp, n2_warp]
            ldrs += [data['ldrs'][p2], data['ldrs'][n1], data['ldrs'][p1], data['ldrs'][n2]]
            for ldr, hdr in zip(ldrs, hdrs):
                inputs.append(torch.cat([ldr, hdr], 1))
        
        net_in = torch.cat(inputs, 1)
        warped_imgs = ({'p2_warp': p2_warp, 'n1_warp': n1_warp, 
                     'p2_warp_hdr': p2_warp_hdr, 'n1_warp_hdr': n1_warp_hdr,
                     'p1_warp': p1_warp, 'n2_warp': n2_warp, 
                     'p1_warp_hdr': p1_warp_hdr, 'n2_warp_hdr': n2_warp_hdr})
        return net_in, merge_hdrs, warped_imgs

    def prepare_fnet_input(self):
        data = self.data
        fnet_in = [self.data['ldrs'][0], self.data['c2p2_adjs'][self.hdr_mid], self.data['ldrs'][3]]
        fnet_in += [self.data['ldrs'][1], self.data['c2p1_adjs'][self.hdr_mid], self.data['ldrs'][4]]
        return fnet_in

    def optimize_weights(self):
        self.loss_terms = {}
        data, pred = self.data, self.pred

        if self.opt['mask_o']:
            roi = 1 - self.data['gt_ref_ws'][self.hdr_mid]
            hdr_loss = self.hdr_crit(self.pred['log_hdr'] * roi, self.data['log_hdrs'][self.hdr_mid] * roi) / (roi.mean() + 1e-8)
            hdr_loss = self.opt['hdr_w'] * hdr_loss
            self.loss_terms['mhdr_loss'] = hdr_loss.item()
            
        else:
            hdr_loss = self.opt['hdr_w'] * self.hdr_crit(self.pred['log_hdr'], self.data['log_hdrs'][self.hdr_mid])
            self.loss_terms['hdr_loss'] = hdr_loss.item()

        self.loss = hdr_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
   
    def _prepare_records(self, data, pred, key=''):
        records = OrderedDict()
        iter_res = []
        hdr_psnr, hdr_ssim = eutils.pt_eval_metrics(data['log_hdr'], pred['log_hdr'].detach())

        hdr_psnr_l, hdr_ssim_l = eutils.pt_eval_metrics(data['hdr'], pred['hdr'].detach())

        if data['log_hdr'].shape[0] > 1: # more than one batch
            records['hdr%s_psnr' % key] = hdr_psnr.mean().item()
            records['hdr%s_ssim' % key] = hdr_ssim.mean().item()

            records['hdr%s_lsnr' % key] = hdr_psnr_l.mean().item()

        h_idx, m_idx, l_idx = data['h_idx'], data['m_idx'], data['l_idx']

        for exp_type, idx in zip(['low', 'mid', 'high'], [l_idx, m_idx, h_idx]):
            psnr = hdr_psnr[idx]
            if len(psnr) > 0:
                records['%s%s_psnr' % (exp_type, key)] = psnr.mean().item() 
                iter_res.append(records['%s%s_psnr' % (exp_type, key)])
                
                psnr_l = hdr_psnr_l[idx]
                records['%s%s_lsnr' % (exp_type, key)] = psnr_l.mean().item() 
                iter_res.append(records['%s%s_lsnr' % (exp_type, key)])

                ssim = hdr_ssim[idx]
                records['%s%s_ssim' % (exp_type, key)] = ssim.mean().item() 
                iter_res.append(records['%s%s_ssim' % (exp_type, key)])

        return records, iter_res

    def prepare_records(self):
        data, pred = self.data, self.pred
        records = OrderedDict()
        iter_res = []

        h_idx, m_idx, l_idx = self.get_expo_level_idxs(data['expos'], self.ldr_mid)
        gt = {'hdr': data['hdrs'][self.hdr_mid], 'log_hdr': data['log_hdrs'][self.hdr_mid], 'h_idx': h_idx, 'm_idx': m_idx, 'l_idx': l_idx}
        pred = {'hdr': self.pred['hdr'], 'log_hdr': self.pred['log_hdr']}
        records, iter_res = self._prepare_records(gt, pred)
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
        visuals = []

        visuals += [data['log_hdrs'][self.hdr_mid], pred['log_hdr']]
        diff = eutils.pt_cal_diff_map(self.pred['log_hdr'].detach(), data['log_hdrs'][self.hdr_mid])
        visuals.append(eutils.pt_colormap(diff))

        if self.opt['mask_o']:
            visuals += [self.data['gt_ref_ws'][self.hdr_mid]]

        for i, ldr in enumerate(data['ldrs']):
            visuals += [ldr]

        visuals.append(eutils.pt_blend_images(data['ldrs']))

        if 'p2_warp' in pred:
            fpred = self.fpred
            flow1_color = eutils.pt_flow_to_color(fpred['flow1'].detach())
            flow2_color = eutils.pt_flow_to_color(fpred['flow2'].detach())
            flow3_color = eutils.pt_flow_to_color(fpred['flow3'].detach())
            flow4_color = eutils.pt_flow_to_color(fpred['flow4'].detach())
            visuals += [flow1_color, flow2_color, flow3_color, flow4_color]

        if 'weights' in self.pred:
            visuals += self.pred['weights']

        if self.split not in ['train', 'val'] and self.opt['origin_hw']:
            new_visuals = eutils.crop_list_of_tensors(visuals, data['hw'])
            return new_visuals
        return visuals
