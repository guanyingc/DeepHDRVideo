"""
Single-stage model (2-exposure)
This model only has a weight net without flow alignment
"""
import os
import torch
import numpy as np
from .base_model import BaseModel
from utils import eval_utils as eutils
from utils import image_utils as iutils
from models import model_utils as mutils
from models import noise_utils as noutils
from models import losses
from collections import OrderedDict
np.random.seed(0)

class hdr2E_model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--mnet_name', default='weight_net', help='mnet means merge net')
        parser.add_argument('--mnet_checkp', default='')
        parser.add_argument('--mnet_afunc', default='LReLU')
        parser.add_argument('--test_root', default='mnet_checkp')
        parser.add_argument('--nframes', type=int, default=3)
        parser.add_argument('--nexps', type=int, default=2)

        parser.add_argument('--tone_low', default=False, action='store_true')
        parser.add_argument('--tone_ref', default=True, action='store_false')

        if is_train:
            parser.set_defaults(init_lr=0.001)

        str_keys = ['mnet_name', 'mnet_afunc']
        val_keys = []
        bool_keys = [] 
        bool_val_dicts = {}
        bool_to_val_dicts = {}
        return parser, str_keys, val_keys, bool_keys, bool_val_dicts, bool_to_val_dicts

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)
        self.net_names = ['mnet']

        c_in, c_out = self.get_io_ch_nums(opt)
        other_opt = {}
        self.mnet = self.import_network(args.mnet_name)(c_in, c_out, c_mid=512, use_bn=opt['use_bn'], 
                afunc=opt['mnet_afunc'], other=other_opt)
        self.mnet = mutils.init_net(self.mnet, init_type=opt['init_type'], gpu_ids=args.gpu_ids)

        if self.is_train: # Criterion
            self.config_optimizers(opt, log)
        self.config_criterions(opt, log)

        self.load_checkpoint(log)

    def get_io_ch_nums(self, opt):
        c_in = c_out = 9
        if self.opt['in_ldr']:
            c_in *= 2
        return c_in, c_out

    def config_criterions(self, opt, log):
        self.hdr_crit = losses.HDRCrit(opt['hdr_loss'], mu=5000) 
        self.ldr_crit = torch.nn.MSELoss()
        self.smooth_crit = losses.SecondOrderSmoothnessLoss(reduction=False, getGrad=True)

        if self.opt['vgg_l'] and self.is_train:
            self.vgg_crit = losses.Vgg16Loss(layers=['relu1_2', 'relu2_2', 'relu3_3'], style_loss=False).to(self.device)


    def config_optimizers(self, opt, log):
        self.optimizer = torch.optim.Adam(self.mnet.parameters(), lr=opt['init_lr'], 
                            betas=(opt['beta_1'], opt['beta_2']))
        self.optimizers.append(self.optimizer)
        self.setup_lr_scheduler() # defined in base model

    def prepare_inputs(self, data, prepare_train=True):
        self.nframes = self.opt['nframes']
        self.nexps = self.opt['nexps']
        self.ldr_mid, self.hdr_mid = self.nframes // 2, (self.nframes - 2) // 2
    
        """ For testing only, speed up image loading by reusing loaded neighboring frames """
        if not self.is_train and self.opt['cached_data'] and data['reuse_cached_data']:
            self.reuse_cached_data(data)
            return

        # Exposure information
        expos = data['expos'].view(-1, self.nframes, 1, 1).split(1, 1)

        # HDR images
        hdrs, log_hdrs = [], []
        for i in range(1, self.nframes-1):
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

        if self.opt['clean_in']: # Depreciated, can be ignored
            clean_l2hdrs = []
            clean_ldrs = []
            for i in range(self.nframes):
                clean_ldrs.append(ldrs[i].clone())
                clean_l2hdrs.append(mutils.pt_ldr_to_hdr(ldrs[i], expos[i]))
            data.update({'clean_ldrs': clean_ldrs, 'clean_l2hdrs': clean_l2hdrs})
        
        # well-exposed mask
        gt_ref_ws = []
        for i in range(1, len(ldrs)-1):
            if self.nexps == 2:
                cur_h_idx = (expos[i] > expos[i-1]).view(-1)
                gt_ref_ws.append(1.0 - self.get_out_mask_method()(ldrs[i], cur_h_idx, h_thr=self.opt['o_hthr'], l_thr=self.opt['o_lthr']))
        
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

        ldr_adjs = []
        for i in range(0, len(ldrs)):
            if i > 0:
                ldr_adjs.append(mutils.pt_ldr_to_ldr(ldrs[i], expos[i], expos[i-1]))
            else:
                ldr_adjs.append(mutils.pt_ldr_to_ldr(ldrs[i], expos[i], expos[i+1]))

        data.update({'hdrs': hdrs, 'log_hdrs': log_hdrs, 'ldrs': ldrs, 'l2hdrs': l2hdrs, 'ldr_adjs': ldr_adjs,
            'expos': expos, 'gt_ref_ws': gt_ref_ws})

    def perturb_low_expo_imgs(self, ldrs, expos):
        need_aug = (self.opt['aug_prob'] == 1.0) or (np.random.uniform() < self.opt['aug_prob'])
        if not need_aug:
            return

        for i in range(self.nexps):
            cur_l_idx = torch.zeros(expos[0].shape[0], device=expos[0].device).byte()
            if i > 0:
               cur_l_idx = cur_l_idx | (expos[i] < expos[i-1]).view(-1)
            if i < self.nframes-1:
               cur_l_idx = cur_l_idx | (expos[i] < expos[i+1]).view(-1)

            if cur_l_idx.sum() > 0:
                tone_d = None
                params = {}
                for j in range(i, self.nframes, self.nexps): # e.g., [0,2], [0,3]
                    if self.opt['tone_low']:
                        ldrs[j][cur_l_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-4, stdv2=1e-3, scale=False)
                    elif self.opt['tone_ref']:
                        ldrs[j][cur_l_idx] = noutils.pt_tone_ref_add_gaussian_noise(ldrs[j][cur_l_idx], stdv1=1e-3, stdv2=1e-3, scale=False)
                    else:
                        raise Exception('Unknown tone low mode')

    def reuse_cached_data(self, data):
        print('Reused cached data')
        ldr_idx, hdr_idx = self.nframes - 1, self.nframes - 2
        
        reused_data_key = ['hdrs', 'log_hdrs', 'ldrs', 'l2hdrs', 'ldr_adjs', 'gt_ref_ws']
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

        cur_h_idx = (expos[hdr_idx] > expos[hdr_idx-1]).view(-1)
        data['gt_ref_ws'].append(1.0 - self.get_out_mask_method()(data['ldrs'][hdr_idx], cur_h_idx, h_thr=self.opt['o_hthr'], l_thr=self.opt['o_lthr']))

        data['l2hdrs'].append(mutils.pt_ldr_to_hdr(ldr, expos[ldr_idx]))

        cur_h_idx = (expos[ldr_idx] > expos[ldr_idx-1]).view(-1)

        data['ldr_adjs'].append(mutils.pt_ldr_to_ldr(ldr, expos[ldr_idx], expos[ldr_idx-1]))

    def get_out_mask_method(self):
        if self.opt['soft_mo']:
            get_out_mask_method = mutils.pt_get_out_blend_mask
        else:
            get_out_mask_method = mutils.pt_ldr_to_1c_mask
        return get_out_mask_method
        
    def forward(self, split='train'):
        self.split = split
        self.prepare_inputs(self.data)

        net_in, merge_hdrs = self.prepare_mnet_inputs(self.opt, self.data) 
        self.pred = self.mnet(net_in, merge_hdrs)
    
        if self.opt['mask_o']:
            mask = self.data['gt_ref_ws'][self.hdr_mid]
            self.pred['hdr'] = self.data['l2hdrs'][self.ldr_mid] * mask + self.pred['hdr'] * (1 - mask)

        self.pred['log_hdr'] = eutils.pt_mulog_transform(self.pred['hdr'], self.mu)

        if not self.is_train:
            self.cached_data = self.data
        self.loss_terms = None
        return self.pred

    def prepare_mnet_inputs(self, opt, data, idxs=[0,1,2]):
        pi, ci, ni = idxs
        prev, cur, nxt = data['ldrs'][pi], data['ldrs'][ci], data['ldrs'][ni]
        prev_hdr, cur_hdr, nxt_hdr = data['l2hdrs'][pi], data['l2hdrs'][ci], data['l2hdrs'][ni]

        inputs = []
        if self.opt['in_ldr']:
            inputs.append(torch.cat([prev, prev_hdr], 1))
            inputs.append(torch.cat([cur, cur_hdr], 1))
            inputs.append(torch.cat([nxt, nxt_hdr], 1))
        else:
            inputs = [prev_hdr, cur_hdr, nxt_hdr]

        net_in = torch.cat(inputs, 1)
        merge_hdrs = [prev_hdr, cur_hdr, nxt_hdr]
        return net_in, merge_hdrs

    def optimize_weights(self):
        self.loss_terms = {}

        if self.opt['mask_o']:
            roi = 1 - self.data['gt_ref_ws'][self.hdr_mid]

            hdr_loss = self.hdr_crit(self.pred['log_hdr'] * roi, self.data['log_hdrs'][self.hdr_mid] * roi) / (roi.mean() + 1e-8)
            hdr_loss = self.opt['hdr_w'] * hdr_loss
            self.loss_terms['mhdr_loss'] = hdr_loss.item()

        else:
            hdr_loss = self.opt['hdr_w'] * self.hdr_crit(self.pred['log_hdr'], self.data['log_hdrs'][self.hdr_mid])
            self.loss_terms['hdr_loss'] = hdr_loss.item()

        self.loss = hdr_loss

        if self.opt['vgg_l']:
            vgg_l, vgg_l_term = self.vgg_crit(self.pred['log_hdr'], self.data['log_hdrs'][self.hdr_mid])
            self.loss += self.opt['vgg_w'] * vgg_l
            for k in vgg_l_term: 
                self.loss_terms[k] = vgg_l_term[k]

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
   
    def prepare_records(self):
        data, pred = self.data, self.pred
        records = OrderedDict()
        iter_res = []

        i = self.hdr_mid
        gt = {'hdr': data['hdrs'][i], 'log_hdr': data['log_hdrs'][i], 'c_expo': data['expos'][i+1], 'p_expo': data['expos'][i]}
        pred = {'hdr': self.pred['hdr'], 'log_hdr': self.pred['log_hdr']}
        records, iter_res = self._prepare_records(gt, pred)

        return records, iter_res

    def _prepare_records(self, data, pred, key=''):
        records = OrderedDict()
        iter_res = []
        hdr_psnr, hdr_ssim = eutils.pt_eval_metrics(data['log_hdr'], pred['log_hdr'].detach())
    
        # PNSR and SSIM in linear HDR domain
        hdr_psnr_l, hdr_ssim_l = eutils.pt_eval_metrics(data['hdr'], pred['hdr'].detach())

        if data['log_hdr'].shape[0] > 1: # more than one batch
            records['hdr%s_psnr' % key] = hdr_psnr.mean().item()
            records['hdr%s_ssim' % key] = hdr_ssim.mean().item()

            records['hdr%s_lsnr' % key] = hdr_psnr_l.mean().item()

        l_idx = data['c_expo'].view(-1) < data['p_expo'].view(-1)
        h_idx = 1 - l_idx

        for exp_type, idx in zip(['low', 'high'], [l_idx, h_idx]):
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
        for i, ldr_adj in enumerate(data['ldr_adjs']):
            visuals += [ldr_adj]

        visuals.append(eutils.pt_blend_images(data['ldrs']))

        if 'weights' in self.pred:
            visuals += self.pred['weights']

        if self.split not in ['train', 'val'] and self.opt['origin_hw']:
            new_visuals = eutils.crop_list_of_tensors(visuals, data['hw'])
            return new_visuals
        return visuals

    def prepare_predict(self):
        if (self.split not in ['train', 'val']) and ('log_hdr_sat' in self.pred):
            prediction = [self.pred['log_hdr_sat']]
        else:
            prediction = [self.pred['log_hdr'].detach()]
        if self.opt['origin_hw']: 
            prediction = eutils.crop_list_of_tensors(prediction, self.data['hw'])
        return prediction 

    def save_visual_details(self, log, split, epoch, i):
        save_dir = log.config_save_detail_dir(split, epoch)
        data, pred = self.data, self.pred
        hdr = pred['hdr']
        if self.opt['origin_hw']: 
            h, w = data['hw']
            hdr = eutils.crop_tensor(hdr, h, w)
        hdr_numpy = hdr[0].cpu().numpy().transpose(1, 2, 0)
        hdr_name = os.path.join(save_dir, '%04d_%s_%s.hdr' % (i, data['scene'][0], data['img_name'][0]))
        iutils.save_hdr(hdr_name, hdr_numpy)
