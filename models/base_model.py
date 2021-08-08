"""
Base model, all models are inherited from this base model
"""
import os
import glob
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from models import model_utils
import importlib

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.gpu_ids = opt['gpu_ids']
        self.device = torch.device('cuda:%d' % (self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.mu = opt['mu'] # for mu-law tone mapping

        self.schedulers = []
        self.net_names = [] # net
        self.fixed_net_names = []
        self.loss_terms = {}
        self.visual_names = []
        self.optimizers = []

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def prepare_inputs(self, data):
        pass

    @abstractmethod
    def forward(self):
        # prepare_inputs()
        # pred = network.forward()
        pass
    
    @abstractmethod
    def optimize_weights(self):
        # optimizer.zero_grad()
        # loss = criterion.forward(pred, data)
        # criterion.backward()
        pass

    @abstractmethod
    def prepare_visual(self):
        pass

    @abstractmethod
    def prepare_records(self):
        pass

    def parse_data(self, sample):
        data = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                data[k] = v.to(self.device)
            else:
                data[k] = v
        self.data = data
        return data

    def get_loss_terms(self):
        return self.loss_terms

    def print_networks(self, log=None):
        for name in self.net_names + self.fixed_net_names:
            network = getattr(self, name)
            print(network)
            log.print_write("=> Parameters of %s: %d" % (name, model_utils.get_params_num(network)))

    def import_network(self, net_name, arch_dir='models.archs', backup_module='networks'):
        print('%s/%s' % (arch_dir.replace('.', '/'), net_name))
        if os.path.exists('%s/%s.py' % (arch_dir.replace('.', '/'), net_name)):
            network_file = importlib.import_module('%s.%s' % (arch_dir, net_name))
        else:
            network_file = importlib.import_module('%s.%s' % (arch_dir, backup_module))
        network = getattr(network_file, net_name)
        return network

    def eval(self):
        for name in self.net_names + self.fixed_net_names:
            print('==>Setting EVAL mode for %s' % name)
            net = getattr(self, name)
            net.eval()

    def train(self):
        for name in self.net_names: # + self.fixed_net_names:
            print('==>Setting TRAIN mode for %s' % name)
            net = getattr(self, name)
            net.train()
    
    def setup_lr_scheduler(self):
        self.schedulers = [model_utils.get_lr_scheduler(self.opt, optimizer) for optimizer in self.optimizers]

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def save_checkpoint(self, epoch, records=None):
        for name in self.net_names:
            network = getattr(self, name)
            if len(self.gpu_ids) > 0:
                state = {'state_dict': network.module.state_dict()}
            else:
                state = {'state_dict': network.state_dict()}
            torch.save(state, os.path.join(self.opt['cp_dir'], '%s_checkp_%0d.pth' % (name, epoch)))
        torch.save(records, os.path.join(self.opt['cp_dir'], 'rec_%0d.pth' % (epoch)))

    def convert_module_checkpoint(self, state_dict):
        new_state_dict = OrderedDict()
        for k in state_dict:
            new_state_dict[k[7:]] = state_dict[k]
        return new_state_dict

    def _load_checkpoint(self, network, checkp_path, log):
        if isinstance(network, torch.nn.DataParallel):
            network = network.module
        checkpoint = torch.load(checkp_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        network.load_state_dict(state_dict, strict=self.opt['strict'])

    def load_checkpoint(self, log):
        for name in self.net_names + self.fixed_net_names:
            if self.opt['checkp_dir'] != '':
                checkps = sorted(glob.glob(self.opt['checkp_dir'] + '/%s_checkp*.pth' % name))
                checkps.sort(key=len)
                checkp = checkps[-1]
                checkp_base = os.path.basename(checkp)
                checkp_suffix = checkp_base[checkp_base.find('_checkp'):]
                checkp_path = os.path.join(self.opt['checkp_dir'], name + checkp_suffix)
            else:
                checkp_path = self.opt['%s_checkp' % name]

            if checkp_path != '':
                log.print_write("==> [%s] loading pre-trained model from %s" % (name, checkp_path))
                network = getattr(self, name)
                self._load_checkpoint(network, checkp_path, log) 

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            #print('%s: does not require gradient!' % str(net))
            net.eval()
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
