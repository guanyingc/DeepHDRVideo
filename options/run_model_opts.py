from .base_opts import BaseOpts
class RunModelOpts(BaseOpts):
    def __init__(self):
        super(RunModelOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Customized
        self.parser.add_argument('--run_model', default=True, action='store_false')
        self.parser.add_argument('--cached_data', default=True, action='store_false', 
                help='reuse loaded neighboring frames in the next iteration during testing')
        self.parser.add_argument('--epochs', default=1, type=int)
        self.parser.add_argument('--save_detail', default=True, action='store_false', help='save .hdr')
        self.parser.add_argument('--result_root', default='doc/experiments')
        self.is_train = False

    def collect_info(self):
        """ the following arguments will be shown in the logdir name """
        self.str_keys += ['model', 'benchmark']
        self.val_keys += ['factor_of_k']
        self.bool_keys += ['cached_data',] #'origin_hw',
        self.bool_val_dicts.update({})
        self.args.str_keys = self.str_keys
        self.args.val_keys = self.val_keys
        self.args.bool_keys = self.bool_keys
        self.args.bool_val_dicts = self.bool_val_dicts
        self.args.bool_to_val_dicts = self.bool_to_val_dicts

    def set_default(self):
        info = ''
        self.args.origin_hw = True # keep original image size
        self.args.suffix = (self.args.suffix + info) if self.args.suffix else info
        self.collect_info()

    def parse(self):
        BaseOpts.parse(self)
        self.set_default()
        return self.args
