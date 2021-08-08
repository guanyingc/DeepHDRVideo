import argparse
import os
import glob
import time
import cv2
import numpy as np
import scipy.io as sio
from skimage import color
color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])

import image_utils as iutils
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--tonemapper', default='Reinhard', help='Reinhard|gamma|log')
    parser.add_argument('-i', '--in_dir', default='')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--max_num', default=-1, type=int)
    parser.add_argument('--tm_gamma', default=0.85, type=float)
    parser.add_argument('--sat_fac', default=1.6, type=float) # saturation factor
    parser.add_argument('--key_fac', default=0.1, type=float) 
    parser.add_argument('--tm_img', default=False, action='store_true')
    parser.add_argument('--img_list', default='')

    parser.add_argument('--out_suffix', default='')
    parser.add_argument('--ext', default='hdr')
    parser.add_argument('--save_ext', default='.png')
    parser.add_argument('--timing', default=False, action='store_true')
    args = parser.parse_args()
    return args

class Tonemapper(object):
    def __init__(self, cfgs):
        assert cfgs['tonemapper'] in ['Reinhard', 'gamma', 'log']
        self.cfgs = cfgs
        self.tm_gamma = cfgs.get('tm_gamma', 0.8500)
        self.color_saturation_fac = cfgs.get('sat_fac', 1.6)
        if cfgs['tonemapper'] in ['Reinhard']:
            self.key_fac = cfgs.get('key_fac', 0.100)
            self.epsilon = 1e-6;

        self.tstart = time.time()

    def load_hdr_list(self, in_dir, img_list='', ext='hdr'):
        print('Input dir: %s\n' % in_dir)
        if img_list != '':
            print('Load HDR List from %s/%s' % (in_dir, img_list))
            hdr_names = np.genfromtxt(os.path.join(in_dir, img_list), 'str')
            hdr_names = [os.path.join(in_dir, name) for name in hdr_names]
            #print(hdr_names)
        else:
            print('Grab hdr files in %s' % in_dir)
            hdr_names = glob.glob(os.path.join(in_dir, '*.%s' % ext))
            hdr_names.sort()
        print('Found %d HDRs' % len(hdr_names))
        return hdr_names

    def config_out_dir(self, in_dir):
        if not os.path.exists(in_dir):
            raise Exception('In dir not exists: %s' % in_dir)
        if self.cfgs['out_suffix'] == '':
            out_suffix = '_%s_PyTM' % (self.cfgs['tonemapper'])

            if self.cfgs['save_ext'] == '.jpg':
                out_suffix += '_JPG'
            elif self.cfgs['save_ext'] == '.png':
                out_suffix += '_PNG'
        else:
            out_suffix = '_%s' % self.cfgs['out_suffix']

        out_dir = utils.remove_slash(in_dir) + out_suffix
        utils.make_file(out_dir)
        self.out_suffix = out_suffix
        return out_dir

    def record_time(self, prompt):
        tend = time.time(); 
        print('%s: %dsec' % (prompt, tend - self.tstart)); 
        self.tstart = tend

    def tonemap_dir(self):
        cfgs = self.cfgs
        in_dir = cfgs['in_dir']
        hdr_list = self.load_hdr_list(in_dir, cfgs['img_list'], cfgs['ext'])

        out_dir = self.config_out_dir(in_dir)
        
        start_i = cfgs.get('start', 0)
        for i in range(start_i, len(hdr_list)):

            if cfgs['max_num'] >= 0 and cfgs['max_num'] <= i:
                break

            hdr_name = hdr_list[i]
            print('Tonemap %d/%d: %s' % (i+1, len(hdr_list), hdr_name))

            save_name = os.path.join(out_dir, '%03d_%s' % (i, os.path.basename(hdr_name)[:-4]))

            ldr = self.tonemap_img(cfgs, save_name, hdr_name, count=i)

            if cfgs['timing']: self.record_time('Tonemapping')
            
            #iutils.save_uint8(save_name + '_noSat.jpg', ldr) # jpg quality -100
            ldr = self.increase_saturation(ldr)
            if cfgs['timing']: self.record_time('Post-processing')

            iutils.save_uint8(save_name + cfgs['save_ext'], ldr)
            if cfgs['timing']: self.record_time('saving')

    def tonemap_img(self, cfgs, save_name, hdr_name, count):
        hdr = iutils.read_hdr(hdr_name)
        print(iutils.get_image_stat(hdr))

        if cfgs['tonemapper'] in ['Reinhard']:
            img_lab = color.rgb2lab(hdr).astype(np.float32)

            if count == cfgs['start'] or cfgs['tm_img']: # not video, no temporal smoothing
                self.log_sum_prev = np.log(self.epsilon + img_lab[:,:,0].reshape(-1)).sum()
                key = self.key_fac
            else:
                h, w, _ = img_lab.shape
                log_sum_cur = np.log(img_lab[:,:,0].reshape(-1) + self.epsilon).sum()
                log_avg_cur = np.exp(log_sum_cur / (h * w))
                log_avg_temp = np.exp((log_sum_cur + self.log_sum_prev) / (2.0 * h * w))

                key = self.key_fac * log_avg_cur / log_avg_temp
                self.log_sum_prev = log_sum_cur

            save_tif_path = save_name + '_TM.tif'
            try:
                command = 'luminance-hdr-cli -l %s --tmo reinhard02 --tmoR02Key %f -g %f --ldrTiff 16b -o %s > /dev/null' % (hdr_name, key, self.tm_gamma, save_tif_path) # New version
                os.system(command)
                ldr = iutils.read_16bit_tif(save_tif_path).astype(np.float32)
                #iutils.save_uint8(save_tif_path[:-4] + '_noSat.jpg', ldr) # jpg quality -100
            except:
                command = 'luminance-hdr-cli -l %s -t reinhard02 -p key=%f -g %f -o %s' % (hdr_name, key, self.tm_gamma, save_tif_path)
                os.system(command)
                ldr = iutils.read_16bit_tif(save_tif_path).astype(np.float32)
            os.remove(save_tif_path)
        elif cfgs['tonemapper'] == 'gamma':
            ldr = np.power(hdr, 1/2.2)
        elif cfgs['tonemapper'] == 'log':
            ldr = iutils.mulog_transform(hdr)
        else:
            raise Exception('Uknown Tonemapper')
        return ldr

    def increase_saturation(self, img, opencv=True):
        if opencv:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * self.color_saturation_fac
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            #img_hsv = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_BGR2HSV)
            #img_hsv[:, :, 1] = img_hsv[:, :, 1] * self.color_saturation_fac
            #img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)[:,:,::-1]
        else:
            img_hsv = color.rgb2hsv(img).astype(np.float32)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * self.color_saturation_fac
            img = color.hsv2rgb(img_hsv)
        return img

def main(args):
    cfgs = vars(args)
    tonemapper = Tonemapper(cfgs)
    tonemapper.tonemap_dir()

if __name__ == '__main__':
    args = parse_args()
    main(args)
