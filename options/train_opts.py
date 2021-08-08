from .base_opts import BaseOpts

class TrainOpts(BaseOpts):
    def __init__(self):
        super(TrainOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Trainining Dataset ####
        self.parser.add_argument('--dataset', default='syn_vimeo_dataset')
        self.parser.add_argument('--data_dir', default='data/vimeo_septuplet', help='For syn_vimeo_dataset')
        self.parser.add_argument('--dataset2', default='syn_hdr_dataset')
        self.parser.add_argument('--hdr_data_dir1', default='data/Synthetic_Train_Data_LiU_HDRv', help='For syn_hdr_dataset')
        self.parser.add_argument('--hdr_data_dir2', default='data/Synthetic_Train_Data_HdM-HDR-2014', help='For syn_hdr_dataset') 
        self.parser.add_argument('--concat', default=True, action='store_false', help='use both dataset and dataset2')
        self.parser.add_argument('--split_train', default='', help='low_only|high_only, for debug') 

        #### Trainining arguments ####
        self.parser.add_argument('--milestones', default=[5, 10, 20], nargs='+', type=int, help='multi-step lr decay')
        self.parser.add_argument('--start_epoch', default=1, type=int)
        self.parser.add_argument('--epochs', default=30, type=int)
        self.parser.add_argument('--batch', default=16, type=int)
        self.parser.add_argument('--val_batch', default=8, type=int)
        self.parser.add_argument('--init_lr', default=0.0001, type=float)
        self.parser.add_argument('--lr_decay', default=0.5, type=float)
        self.parser.add_argument('--beta_1', default=0.9, type=float, help='adam')
        self.parser.add_argument('--beta_2', default=0.999, type=float, help='adam')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='sgd')
        self.parser.add_argument('--w_decay', default=4e-4, type=float)

        self.is_train = True

    def collect_info(self): 
        BaseOpts.collect_info(self)
        """ the following arguments will be shown in the logdir name """
        self.str_keys += ['img_loss', 'split_train',
                ]
        self.val_keys  += ['batch', 'init_lr',
                ]
        self.bool_keys += ['vgg_l', 'concat',
                ] 
        self.bool_to_val_dicts.update({'vgg_l': 'vgg_w'})
        self.args.str_keys = self.str_keys
        self.args.val_keys = self.val_keys
        self.args.bool_keys = self.bool_keys
        self.args.bool_val_dicts = self.bool_val_dicts
        self.args.bool_to_val_dicts = self.bool_to_val_dicts

    def set_default(self):
        BaseOpts.set_default(self)
        self.collect_info()

    def parse(self):
        BaseOpts.parse(self)
        self.set_default()
        return self.args
