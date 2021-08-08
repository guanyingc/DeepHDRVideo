""" 
Logger for saving results and visualizations
Some functions might be redundant
"""

import datetime
import time
import os
import numpy as np
import torch
import torchvision.utils as vutils
import utils.eval_utils as eutils
from . import utils

""" seting matplotlib for plotting"""
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
plt.rcParams["figure.figsize"] = (5,8)

class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        if args.make_dir:
            self._setup_dirs(args)
        self.args = args
        self.print_args()

    #### Directory related
    def _parse_arguments(self, args):
        info = ''
        arg_var  = vars(args)
        if hasattr(args, 'run_model') and args.run_model:
            test_root = arg_var['test_root']
            info += ',%s' % os.path.basename(arg_var[test_root]).split('.')[0]
        for k in args.str_keys:  
            if arg_var[k] != '':
                info = '{0},{1}'.format(info, arg_var[k])
        for k in args.val_keys:  
            var_key = k[:2] + '_' + k[-1] if len(k) > 3 else k
            info = '{0},{1}-{2}'.format(info, var_key, arg_var[k])
        for k in args.bool_keys: 
            if arg_var[k]:
                if k in args.bool_to_val_dicts.keys():
                    val_key = args.bool_to_val_dicts[k]
                    save_key = val_key[:2] + '_' + val_key[-1]
                    info = '{0},{1}-{2}'.format(info, save_key, arg_var[val_key])
                elif k in args.bool_val_dicts.keys():
                    val_key = args.bool_val_dicts[k]
                    save_key = val_key[:2] + '_' + val_key[-1]
                    info = '{0},{1},{2}-{3}'.format(info, k, save_key, arg_var[val_key])
                else:
                    info = '{0},{1}'.format(info, k) #if arg_var[k] else info 
        return info

    def _setup_dirs(self, args):
        date_now = datetime.datetime.now()
        self.date = '%02d-%02d' % (date_now.month, date_now.day)
        dir_name = self.date
        dir_name += (',%s' % args.suffix) if args.suffix else ''
        dir_name += self._parse_arguments(args) 
        dir_name += ',DEBUG' if args.debug else ''

        self._check_path(args, dir_name)
        file_dir = os.path.join(args.log_dir, '%s,%s' % (dir_name, date_now.strftime('%H:%M:%S')))
        self.log_fie = open(file_dir, 'w')
        return 

    def _check_path(self, args, dir_name):
        if hasattr(args, 'run_model') and args.run_model:
            test_root = vars(args)[args.test_root]
            data_dir = os.path.basename(os.path.normpath(args.bm_dir))
            if len(data_dir) > 30:
                data_dir = data_dir[:10] + data_dir[-10:]
            if args.checkp_dir != '':
                log_root = os.path.join(args.checkp_dir, dir_name + ',%s' % data_dir)
            else:
                log_root = os.path.join(os.path.dirname(test_root), dir_name + ',%s' % data_dir)
            args.log_dir = log_root
            sub_dirs = ['test']
        else:
            if args.debug:
                dir_name = 'DEBUG/' + dir_name
            log_root = os.path.join(args.save_root, args.dataset, args.item, dir_name)
            args.log_dir = os.path.join(log_root)
            args.cp_dir  = os.path.join(log_root, 'checkpointdir')
            utils.make_files([args.log_dir, args.cp_dir])
            sub_dirs = ['train', 'val', 'test']

        for sub_dir in sub_dirs:
            utils.make_files([os.path.join(args.log_dir, sub_dir)])

    #### Print related
    def get_time_info(self, epoch, iters, batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0
        total_iters  = (self.args.epochs - self.args.start_epoch + 1) * batch
        cur_iters    = (epoch - self.args.start_epoch) * batch + iters
        time_total   = time_elapsed * (float(total_iters) / cur_iters)
        return time_elapsed, time_total

    def print_write(self, strs):
        print('%s' % strs)
        if self.args.make_dir:
            self.log_fie.write('%s\n' % strs)
            self.log_fie.flush()

    def print_args(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(utils.dict_to_string(vars(self.args)))
        strs += '-------------- End ----------------\n'
        self.print_write(strs)

    def print_iters_summary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)

        if opt['split'] == 'train': 
            time_elapsed, time_total = self.get_time_info(epoch, iters, batch) # Buggy for test
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        self.print_write(strs)

        if 'timer' in opt.keys(): 
            self.print_write(opt['timer'].time_to_string())

        if 'recorder' in opt.keys(): 
            self.print_write(opt['recorder'].iter_rec_to_string(opt['split'], epoch))

    def print_epoch_summary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        self.print_write('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        summary_str = opt['recorder'].epoch_rec_to_string(split, epoch)
        self.print_write(summary_str)
        return summary_str
    
    #### Image processing related
    def convert_to_same_size(self, t_list):
        shape = (t_list[0].shape[0], 3, t_list[0].shape[2], t_list[0].shape[3])
        for i, tensor in enumerate(t_list):
            n, c, h, w = tensor.shape
            if tensor.shape[1] != shape[1]: # check channel
                if tensor.shape[1] == 2:
                    new_tensor = torch.zeros(n, 3, h, w, device=tensor.device)
                    new_tensor[:,:2] = tensor
                    tensor = new_tensor
                else: # c == 3
                    tensor = tensor.expand((n, shape[1], h, w))
                t_list[i] = tensor
            if h == shape[2] and w == shape[3]:
                continue
            t_list[i] = torch.nn.functional.interpolate(tensor, [shape[2], shape[3]], mode='bilinear', align_corners=False)
        return t_list

    def split_multi_channel(self, t_list, max_save_n = 8):
        new_list = []
        for tensor in t_list:
            if tensor.shape[1] > 3:
                num = 3
                new_list += torch.split(tensor, num, 1)[:max_save_n]
            else:
                new_list.append(tensor)
        return new_list

    #### Result saving related
    def get_save_dir(self, split, epoch):
        save_dir = os.path.join(self.args.log_dir, split)
        run_model = hasattr(self.args, 'run_model') and self.args.run_model
        if not run_model and epoch > 0:
            save_dir = os.path.join(save_dir, '%02d' % (epoch))
        utils.make_file(save_dir)
        return save_dir

    def save_separate(self, res, split, epoch, iters, idx=0, save_dir=''):
        if save_dir == '':
            save_dir = os.path.join(self.get_save_dir(split, epoch))
        n, c, h, w = res.shape
        for i in range(n):
            vutils.save_image(res[i], os.path.join(save_dir, 'pred_%02d_%05d_%02d_%02d.jpg' % (epoch, iters, i, idx)))

    def config_save_detail_dir(self, split, epoch, subdir='Details'):
        save_detail_dir = os.path.join(self.args.log_dir, split, subdir, '%02d' % epoch)
        utils.make_file(save_detail_dir)
        return save_detail_dir

    def save_feats(self, feats, split, epoch, iters, clip=True):
        save_dir = self.config_save_detail_dir(split, epoch, 'feats')
        for i, feat in enumerate(feats):
            feat = feat[0] #.unsqueeze(1) # only save the first batch
            max_feat = feat.max()
            if clip:
                feat_normalized = (feat / max_feat).clamp(0, 1)
            else:
                feat_normalized = ((feat/ max_feat).clamp(-1, 1) + 1) / 2.0 #[0, 0.5], [0.5, 1]
            feat_color = eutils.pt_colormap(feat_normalized, thres=1)
            #feat_split = torch.split(feat_color, 1, 0)
            save_name = os.path.join(save_dir, 'feat_%02d_%05d_%02d.jpg' % (epoch, iters, i))
            print(save_name)
            vutils.save_image(list(feat_color), save_name, nrow=16)

    def save_img_results(self, results, split, epoch, iters, nrow, error=''):
        max_save_n = self.args.test_save_n if split == 'test' else self.args.train_save_n
        res = [img.detach().cpu() for img in results]
        res = self.split_multi_channel(res, max_save_n)
        res = torch.cat(self.convert_to_same_size(res), 0)
        save_dir = self.get_save_dir(split, epoch)
        save_prefix = os.path.join(save_dir, '%02d_%05d' % (epoch, iters))
        save_prefix += ('_%s' % error) if error != '' else ''
        #if self.args.save_separate: 
        #    self.save_separate(res, split, epoch, iters)
        #else:
        vutils.save_image(res, save_prefix + '_out.jpg', nrow=nrow, pad_value=1)
    
    def save_txt_result(self, results, split, epoch, summary_str):
        max_row_len = 0
        for i in range(len(results)):
            if len(results[i]) > max_row_len: max_row_len = len(results[i])
        for i in range(len(results)):
            if len(results[i]) < max_row_len: results[i] += [0] * (max_row_len - len(results[i]))
                
        mean_res = np.array(results).mean(0)
        res = np.vstack([np.array(results), mean_res])
        save_name = '%s_%s_%d_res.txt' % (self.args.suffix, split, epoch)
        np.savetxt(os.path.join(self.args.log_dir, split, save_name), res, fmt='%.3f')

        if res.ndim > 1:
            for i in range(res.shape[1]):
                save_name = '%s_%d_res.txt' % (self.args.suffix, i)
                np.savetxt(os.path.join(self.args.log_dir, split, save_name), res[:,i], fmt='%.3f')

        if hasattr(self.args, 'run_model') and self.args.run_model:
            self.save_summary_result(mean_res, summary_str)

    def save_summary_result(self, mean_res, summary_str):
        save_dir = self.args.result_root
        utils.make_file(save_dir)
        import socket
        hostname = socket.gethostname()[:7]
        save_file = os.path.join(self.args.result_root, '%s_%s_%s.txt' % (
                self.date, hostname, self.args.suffix))
        summary_file = open(save_file, 'a+')
        summary_file.write('%s\n' % self.args.log_dir)
        summary_file.write('%s\n' % self.args.suffix)
        results = '\n'.join(['%.3f' % value for value in mean_res])
        summary_file.write(results + '\n')
        summary_file.write(summary_str + '\n')
        summary_file.close()
         
    #### Plotting related
    def plot_all_curves(self, recorder, split='train', epoch=-1, intv=1):
        save_dir = self.args.log_dir
        save_name = 'Summary.jpg' 
        records = recorder.records
        all_classes = recorder.classes
        splits = records.keys() 
        classes = []
        for split in splits:
            for c in all_classes + ['lr']:
                for k in records[split].keys():
                    #print(split, k, c)
                    if c in k.lower() and c not in classes:
                        classes.append(c)
        #print(classes)
        if len(classes) == 0: return

        fig, axes = plt.subplots(len(classes), len(splits), figsize=(5*len(splits), 4*len(classes))) # W, H
        if len(splits) == 1: axes = axes.reshape(-1, 1)
        if len(classes) == 1: axes = axes.reshape(1, -1)
        for idx_s, split in enumerate(splits):
            dict_of_array = recorder.record_to_dict_of_array(split, epoch, intv)
            for idx, c in enumerate(classes):
                ax = axes[idx][idx_s]
                ax.grid()
                legends = []
                for k in dict_of_array.keys():
                    if (c in k.lower()) and not k.endswith('_x'):
                        ax.plot(dict_of_array[k+'_x'], dict_of_array[k])
                        legends.append(k)
                if len(legends) != 0:
                    ax.legend(legends, bbox_to_anchor=(0.5, -0.05), loc='upper center', 
                                ncol=3, prop=fontP)
                    ax.set_title(c)
                    #ax.set_xlabel('Epoch') 
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.clf()
        plt.close(fig)

    def plot_curves(self, recorder, split='train', epoch=-1, intv=1):
        dict_of_array = recorder.record_to_dict_of_array(split, epoch, intv)
        save_dir = os.path.join(self.args.log_dir, split)
        if epoch < 0:
            save_dir = self.args.log_dir
            save_name = '%s_Summary.jpg' % (split)
        else:
            save_name = '%s_epoch_%d.jpg' % (split, epoch)

        classes = recorder.classes
        classes = utils.check_in_list(classes, dict_of_array.keys())
        if len(classes) == 0: return

        fig, axes = plt.subplots(len(classes), 1, figsize=(5, 4*len(classes))) # W, H
        if len(classes) == 1:
            axes = [axes]
        for idx, c in enumerate(classes):
            ax = axes[idx]
            #plt.subplot(len(classes), 1, idx+1)
            ax.grid()
            legends = []
            for k in dict_of_array.keys():
                if (c in k.lower()) and not k.endswith('_x'):
                    ax.plot(dict_of_array[k+'_x'], dict_of_array[k])
                    legends.append(k)
            if len(legends) != 0:
                ax.legend(legends, bbox_to_anchor=(0.5, -0.05), loc='upper center', 
                            ncol=3, prop=fontP)
                ax.set_title(c)
                #if epoch < 0: plt.xlabel('Epoch') 
                #else: plt.xlabel('Iters')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.clf()
        plt.close(fig)

