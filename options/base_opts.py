import argparse
import models
import datasets
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # lists to store important parameters that will be displayed in the logfile name
        self.str_keys = [] 
        self.val_keys = []
        self.bool_keys = []
        self.bool_val_dicts = {}
        self.bool_to_val_dicts = {}

    def initialize(self):
        #### Training Data Preprocessing Arguments ####
        self.parser.add_argument('--rescale', default=True, action='store_false')
        self.parser.add_argument('--sc_k', default=1.0, type=float, help='scale factor')
        self.parser.add_argument('--rand_sc', default=True, action='store_false')
        self.parser.add_argument('--scale_h', default=320, type=int)
        self.parser.add_argument('--scale_w', default=320, type=int)
        self.parser.add_argument('--crop', default=True, action='store_false')
        self.parser.add_argument('--crop_h', default=256, type=int)
        self.parser.add_argument('--crop_w', default=256, type=int)
        self.parser.add_argument('--tone_d', default=0.1, type=float) # TODO
        self.parser.add_argument('--aug_prob', default=0.9, type=float, help='prob of perturbing tone of input')
        self.parser.add_argument('--cl_aug', default=True, action='store_false', help='color augmentation')
        self.parser.add_argument('--rotate90', default=True, action='store_false')
        self.parser.add_argument('--rotate_aug', default=False, action='store_true')
        self.parser.add_argument('--flip_aug', default=True, action='store_false')
        self.parser.add_argument('--inv_aug', default=True, action='store_false', help='inverse time order')
        self.parser.add_argument('--rstop', default=True, action='store_false', help='random stop value for high exposure')
        self.parser.add_argument('--repeat', default=1, type=int, help='repeat dataset in an epoch')
        self.parser.add_argument('--static', default=False, action='store_true', help='input static frames, for debug')
        self.parser.add_argument('--clean_in', default=False, action='store_true') # TODO

        #### Device Arguments ####
        self.parser.add_argument('--gpu_ids', default='0', help='0,1,2,3 for gpu / -1 for cpu')
        self.parser.add_argument('--time_sync', default=True, action='store_false')
        self.parser.add_argument('--workers', default=8, type=int)
        self.parser.add_argument('--seed', default=0, type=int)

        #### Network Arguments ####
        self.parser.add_argument('--model', default='hdr2E_model')
        self.parser.add_argument('--rb_cnum', default=256, type=int) # residual block channel num
        self.parser.add_argument('--rb_lnum', default=12, type=int) # residual block layer num
        self.parser.add_argument('--mask_o', default=False, action='store_true')
        self.parser.add_argument('--soft_mo', default=False, action='store_true')
        self.parser.add_argument('--lthr', default=0.15, type=float) # low threshold for in mask
        self.parser.add_argument('--hthr', default=0.9, type=float) # high threshold for in mask
        self.parser.add_argument('--o_lthr', default=0.15, type=float) # output thres for well-exposed regions
        self.parser.add_argument('--o_hthr', default=0.9, type=float) # output thres for well-exposed regions
        self.parser.add_argument('--checkp_dir', default='')
        self.parser.add_argument('--strict', default=True, action='store_false', help='strict matching for checkpoint')
        self.parser.add_argument('--mu', default=5000, type=float, help='for mulog tone mapping')

        self.parser.add_argument('--up', default='deconv', help='nearest,deconv')
        self.parser.add_argument('--down', default='conv', help='max,avg')
        self.parser.add_argument('--init_type', default='kaiming')
        self.parser.add_argument('--use_bn', default=True, action='store_false')
        self.parser.add_argument('--save_intv', default=2, type=int, help='interval of saving model checkpoint')
        self.parser.add_argument('--in_ldr', default=True, action='store_false',
                help='if taking the ldr images as input for the weight net')
        self.parser.add_argument('--m_in_nb', default=True, action='store_false',
                help='if taking the neigboring frames as input for the weight net')

        #### Loss Arguments ####
        self.parser.add_argument('--vgg_l', default=False, action='store_true')
        self.parser.add_argument('--vgg_w', default=1, type=float)
        self.parser.add_argument('--img_loss', default='l1', help='l1|l2')
        self.parser.add_argument('--img_w', default=1, type=float)
        self.parser.add_argument('--hdr_loss', default='l1', help='l1|l2')
        self.parser.add_argument('--hdr_w', default=1, type=float)

        #### Displaying Arguments ####
        self.parser.add_argument('--train_disp', default=20, type=int, help='print interval in training')
        self.parser.add_argument('--train_save', default=400, type=int, help='save visualization interval in training')
        self.parser.add_argument('--val_intv', default=1, type=int, help='validation interval')
        self.parser.add_argument('--val_disp', default=1, type=int)
        self.parser.add_argument('--val_save', default=10, type=int)
        self.parser.add_argument('--max_train_iter',default=-1, type=int, help='max iters in an epoch during training')
        self.parser.add_argument('--max_val_iter', default=-1, type=int)
        self.parser.add_argument('--train_save_n', default=4, type=int, help='for saving visualizaton')
        self.parser.add_argument('--test_save_n', default=4, type=int, help='for saving visualizaton')

        self.parser.add_argument('--test_batch', default=1, type=int) # TODO: clean
        self.parser.add_argument('--test_disp', default=1, type=int)
        self.parser.add_argument('--test_save', default=1, type=int)

        #### Testing Dataset Arguments #### 
        self.parser.add_argument('--benchmark', default='tog13_online_align_dataset')
        self.parser.add_argument('--bm_dir', default='data/TOG13_Dynamic_Dataset')
        self.parser.add_argument('--test_scene', default=None)
        self.parser.add_argument('--max_test_iter', default=-1, type=int)
        self.parser.add_argument('--disable_save', default=False, action='store_true')
        self.parser.add_argument('--save_records', default=True, action='store_false')
        self.parser.add_argument('--skip_headtail', default=True, action='store_false', help='skip the first and last frame')
        self.parser.add_argument('--align', default=False, action='store_true', help='perform only alignment')

        self.parser.add_argument('--test_tog13_frames', default=-1, type=int, help='max test frame number for tog13')
        self.parser.add_argument('--test_real_frames', default=-1, type=int, help='max test frames for real dataset')
        self.parser.add_argument('--test_syn_frames', default=-1, type=int, help='max test frames for syn test data')
        self.parser.add_argument('--factor_of_k', default=8, type=int, help='keep image size to be factor of k')
        self.parser.add_argument('--origin_hw', default=False, action='store_true')
        self.parser.add_argument('--start_idx', default=0, type=int, help='start frame index when testing on real data')
        self.parser.add_argument('--test_resc', default=False, action='store_true')
        self.parser.add_argument('--test_h', default=720, type=int)
        self.parser.add_argument('--test_w', default=1280, type=int)

        #### Log Arguments ####
        self.parser.add_argument('--save_root', default='logdir/')
        self.parser.add_argument('--item', default='ICCV', help='specify the log subdir')
        self.parser.add_argument('--suffix', default=None, help='will appear in the logdir name')
        self.parser.add_argument('--debug', default=False, action='store_true', help='enable debug mode')
        self.parser.add_argument('--make_dir', default=True, action='store_false')

    def set_default(self):
        if self.args.debug: # debug mode
            self.args.train_disp = 1
            self.args.train_save = 1
            self.args.val_save = 1
            self.args.max_train_iter = 4 
            self.args.max_val_iter = 4

    def collect_info(self):
        """ the following arguments will be shown in the logdir name """
        self.str_keys += ['model', 'init_type'
                ]
        self.val_keys += ['crop_h', 'hthr'
                ]
        self.bool_keys += ['use_bn', 'in_ldr', 'rescale', 'align', 
                ] 
        self.bool_val_dicts.update({})
        self.bool_to_val_dicts.update({'rescale': 'scale_h'})

    def gather_options(self):
        args, _ = self.parser.parse_known_args()
        
        # Update model-related parser options
        modelOptionSetter = models.get_option_setter(args.model)
        parser, str_keys, val_keys, bool_keys, bool_val_dicts, bool_to_val_dicts = modelOptionSetter(self.parser, self.is_train)
        args, _ = parser.parse_known_args()

        # Update dataset-specific parser options
        if self.is_train:
            data_option_setter = datasets.get_option_setter(args.dataset)
        else:
            data_option_setter = datasets.get_option_setter(args.benchmark)
        parser = data_option_setter(self.parser, self.is_train)
        self.parser = parser

        self.str_keys += str_keys
        self.val_keys += val_keys
        self.bool_keys += bool_keys
        self.bool_val_dicts.update(bool_val_dicts)
        self.bool_to_val_dicts.update(bool_to_val_dicts)
        return parser.parse_args()

    def parse(self):
        args = self.gather_options()
        args.is_train = self.is_train

        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])
        self.args = args
        return self.args
