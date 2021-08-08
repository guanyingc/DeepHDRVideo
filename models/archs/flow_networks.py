"""
Network architecture for flow net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_utils as mutils
from models import network_utils as nutils
from collections import OrderedDict
import numpy as np

# flow Spynet
class spynet_triple(nn.Module):
    """ for 2-exposure """
    def __init__(self, c_in, c_out, requires_grad=True, other={}):
        super(spynet_triple, self).__init__()

        class Basic(torch.nn.Module):
            def __init__(self, intLevel, c_in, c_out, afunc='ReLU'):
                super(Basic, self).__init__()
                afunc = nutils.activation(afunc)
                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(c_in, 100, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(100, 50, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(50, 25, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(25, c_out, kernel_size=5, stride=1, padding=2),
            )

            def forward(self, inputs):
                flows = self.moduleBasic(inputs).clamp(-320, 320)
                flow1 = flows.narrow(1, 0, 2)
                flow2 = flows.narrow(1, 2, 2)
                return flow1, flow2

        self.scales = 5
        self.down_factor = 2**4 # 16
        self.share = other.get('fshare', False)
        if self.share:
            print('*** Spynet: Share weights for different layers**')
            self.moduleBasic = Basic(0, c_in=c_in, c_out=c_out, afunc='ReLU')
        else:
            self.moduleBasic = torch.nn.ModuleList([Basic(
                i, c_in=c_in, c_out=c_out, afunc='ReLU') for i in range(self.scales) ])

        self.backward_grid = {}
        if not requires_grad:
            print('Spynet does not require grads!')
            for param in self.parameters():
                param.requires_grad = False

    def prepare_inputs(self, data):
        prev, cur, nxt = data
        n, c, h, w = prev.shape
        self.resized_img = False
        if not mutils.check_valid_input_size(self.down_factor, h, w):
            self.resized_img = True
            self.in_h, self.in_w = h, w
            prev = mutils.resize_img_to_factor_of_k(prev, k=self.down_factor)
            cur = mutils.resize_img_to_factor_of_k(cur, k=self.down_factor)
            nxt = mutils.resize_img_to_factor_of_k(nxt, k=self.down_factor)
            #print('Flownet Resizing img from %dx%d to %dx%d' % (h, w, cur.shape[2], cur.shape[3]))

        prev_pyramid, cur_pyramid, nxt_pyramid = [prev], [cur], [nxt]
        for i in range(self.scales - 1):
            prev_pyramid.insert(0, torch.nn.functional.avg_pool2d(prev_pyramid[0], 2))
            cur_pyramid.insert(0, torch.nn.functional.avg_pool2d(cur_pyramid[0], 2))
            nxt_pyramid.insert(0, torch.nn.functional.avg_pool2d(nxt_pyramid[0], 2))
        return prev_pyramid, cur_pyramid, nxt_pyramid

    def forward(self, data):
        prev_pyramid, cur_pyramid, nxt_pyramid = self.prepare_inputs(data)

        for i in range(len(prev_pyramid)):
            #print('Estimating %d level flow: %s' % (i, str(prev_pyramid[i].shape)))
            if i == 0:
                p, n = prev_pyramid[i], nxt_pyramid[i]
            else:
                shape = prev_pyramid[i].shape
                grid_key = str(prev_pyramid[i].device) + '_' + str(shape)
                if grid_key not in self.backward_grid:
                    self.backward_grid[grid_key] = nutils.generate_grid(prev_pyramid[i])
                # upsample
                up_flow1 = torch.nn.functional.interpolate(flow1, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                up_flow2 = torch.nn.functional.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                p = nutils.backward_warp(prev_pyramid[i], up_flow1, self.backward_grid[grid_key])
                n = nutils.backward_warp(nxt_pyramid[i], up_flow2, self.backward_grid[grid_key])
            input = torch.cat([p, cur_pyramid[i], n], 1)
            if self.share:
                flow1, flow2 = self.moduleBasic(input)
            else:
                flow1, flow2 = self.moduleBasic[i](input)
            if i > 0:
                flow1 += up_flow1
                flow2 += up_flow2

        pred = OrderedDict()
        if self.resized_img:
            flow1 = mutils.resize_flow(flow1, self.in_h, self.in_w)
            flow2 = mutils.resize_flow(flow2, self.in_h, self.in_w)
        pred['flow1'] = flow1.clamp(-200, 200)
        pred['flow2'] = flow2.clamp(-200, 200)
        return pred

class spynet_2triple(nn.Module):
    """ for 3-exposure """
    def __init__(self, c_in, c_out, requires_grad=True, other={}):
        super(spynet_2triple, self).__init__()

        class Basic(torch.nn.Module):
            def __init__(self, intLevel, c_in, c_out, afunc='ReLU'):
                super(Basic, self).__init__()
                afunc = nutils.activation(afunc)
                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(c_in, 100, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(100, 50, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(50, 25, kernel_size=5, stride=1, padding=2),
                    afunc,
                    torch.nn.Conv2d(25, c_out, kernel_size=5, stride=1, padding=2),
            )

            def forward(self, inputs):
                flows = self.moduleBasic(inputs).clamp(-320, 320)
                flow1 = flows.narrow(1, 0, 2)
                flow2 = flows.narrow(1, 2, 2)
                flow3 = flows.narrow(1, 4, 2)
                flow4 = flows.narrow(1, 6, 2)
                return flow1, flow2, flow3, flow4

        self.scales = 5
        self.down_factor = 2**4 # 16
        self.share = other.get('fshare', False)
        if self.share:
            print('*** Spynet: Share weights for different layers**')
            self.moduleBasic = Basic(0, c_in=c_in, c_out=c_out, afunc='ReLU')
        else:
            self.moduleBasic = torch.nn.ModuleList([Basic(
                i, c_in=c_in, c_out=c_out, afunc='ReLU') for i in range(self.scales) ])

        self.backward_grid = {}
        if not requires_grad:
            print('Spynet does not require grads!')
            for param in self.parameters():
                param.requires_grad = False

    def prepare_inputs(self, data):
        p2, c_adj_p2, n1, p1, c_adj_p1, n2 = data

        n, c, h, w = p2.shape
        self.resized_img = False
        if not mutils.check_valid_input_size(self.down_factor, h, w):
            self.resized_img = True
            self.in_h, self.in_w = h, w
            p2 = mutils.resize_img_to_factor_of_k(p2, k=self.down_factor)
            c_adj_p2 = mutils.resize_img_to_factor_of_k(c_adj_p2, k=self.down_factor)
            n1 = mutils.resize_img_to_factor_of_k(n1, k=self.down_factor)
            p1 = mutils.resize_img_to_factor_of_k(p1, k=self.down_factor)
            c_adj_p1 = mutils.resize_img_to_factor_of_k(c_adj_p1, k=self.down_factor)
            n2 = mutils.resize_img_to_factor_of_k(n2, k=self.down_factor)
            #print('Flownet Resizing img from %dx%d to %dx%d' % (h, w, p2.shape[2], p2.shape[3]))

        #prev, cur, nxt = data
        p2_prm, c_adj_p2_prm, n1_prm = [p2], [c_adj_p2], [n1]
        p1_prm, c_adj_p1_prm, n2_prm = [p1], [c_adj_p1], [n2]
        for i in range(self.scales - 1):
            p2_prm.insert(0, F.avg_pool2d(p2_prm[0], 2))
            c_adj_p2_prm.insert(0, F.avg_pool2d(c_adj_p2_prm[0], 2))
            n1_prm.insert(0, F.avg_pool2d(n1_prm[0], 2))

            p1_prm.insert(0, F.avg_pool2d(p1_prm[0], 2))
            c_adj_p1_prm.insert(0, F.avg_pool2d(c_adj_p1_prm[0], 2))
            n2_prm.insert(0, F.avg_pool2d(n2_prm[0], 2))
        return p2_prm, c_adj_p2_prm, n1_prm, p1_prm, c_adj_p1_prm, n2_prm

    def forward(self, data):
        p2_prm, c_adj_p2_prm, n1_prm, p1_prm, c_adj_p1_prm, n2_prm = self.prepare_inputs(data)

        for i in range(len(p2_prm)):
            #print('Estimating %d level flow: %s' % (i, str(prev_pyramid[i].shape)))
            if i == 0:
                p2, n1 = p2_prm[i], n1_prm[i]
                p1, n2 = p1_prm[i], n2_prm[i]
            else:
                shape = p2_prm[i].shape
                grid_key = str(p2_prm[i].device) + '_' + str(shape)
                if grid_key not in self.backward_grid:
                    self.backward_grid[grid_key] = nutils.generate_grid(p2_prm[i])

                # upsample
                up_flow1 = F.interpolate(flow1, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                up_flow2 = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                up_flow3 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                up_flow4 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                p2 = nutils.backward_warp(p2_prm[i], up_flow1, self.backward_grid[grid_key])
                n1 = nutils.backward_warp(n1_prm[i], up_flow2, self.backward_grid[grid_key])

                p1 = nutils.backward_warp(p1_prm[i], up_flow3, self.backward_grid[grid_key])
                n2 = nutils.backward_warp(n2_prm[i], up_flow4, self.backward_grid[grid_key])
            input = torch.cat([p2, c_adj_p2_prm[i], n1, p1, c_adj_p1_prm[i], n2], 1)
            if self.share:
                flow1, flow2, flow3, flow4 = self.moduleBasic(input)
            else:
                flow1, flow2, flow3, flow4 = self.moduleBasic[i](input)
            if i > 0:
                flow1 += up_flow1
                flow2 += up_flow2
                flow3 += up_flow3
                flow4 += up_flow4
        pred = OrderedDict()
        if self.resized_img:
            flow1 = mutils.resize_flow(flow1, self.in_h, self.in_w)
            flow2 = mutils.resize_flow(flow2, self.in_h, self.in_w)
            flow3 = mutils.resize_flow(flow3, self.in_h, self.in_w)
            flow4 = mutils.resize_flow(flow4, self.in_h, self.in_w)
        pred['flow1'] = flow1.clamp(-200, 200)
        pred['flow2'] = flow2.clamp(-200, 200)
        pred['flow3'] = flow3.clamp(-200, 200)
        pred['flow4'] = flow4.clamp(-200, 200)
        return pred

