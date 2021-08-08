"""
Loss functions
Many function are not used in this repository
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from utils import eval_utils
from models import model_utils as mutils

# Image space loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class Charbonnier(nn.Module):
    def __init__(self, eps=1e-6, reduction=True):
        super(Charbonnier, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x):
        #num = x.nelement()
        error = torch.sqrt(x * x + self.eps)
        if self.reduction:
            error = torch.mean(torch.sqrt(diff))
        return error

def gradient(tensor):
    dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    return dx, dy

def edge_aware_smooth_loss(img, flow):
    N, C, H, W = flow.shape
    img_dx, img_dy = gradient(img.mean(1, keepdim=True))
    flow_dx, flow_dy = gradient(flow)
    w_dx = torch.exp(-img_dx * img_dx) # [B, 1, H, W] in [0, 1]
    w_dy = torch.exp(-img_dy * img_dy)
    loss = w_dx
    print('w_dx', w_dx.max(), w_dx.min())

class SmoothnessLoss(nn.Module):
    def __init__(self, cs=20.0, reduction=True, getGrad=False):
        super(SmoothnessLoss, self).__init__()
        self.func = Charbonnier(reduction=False)
        self.reduction = reduction
        self.getGrad = getGrad
        self.cs = cs

    def forward(self, input, img):
        idx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean(1, True) # image gradient
        idy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean(1, True)
        dx = input[:, :, :, 1:] - input[:, :, :, :-1] # estimation gradient
        dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        wx = torch.exp(-self.cs * idx)
        wy = torch.exp(-self.cs * idy)
        loss_x = wx * self.func(dx) 
        loss_y = wy * self.func(dy)
        if self.reduction:
            loss = loss_x.mean() + loss_y.mean()
        else:
            loss = loss_x[:, :, :-1, :] + loss_y[:, :, :, :-1]
        if self.getGrad:
            return loss, idx[:, :, :-1, :] + idy[:, :, :, :-1]
        else:
            return loss

class SecondOrderSmoothnessLoss(nn.Module):
    def __init__(self, cs=20.0, reduction=True, getGrad=False):
        super(SecondOrderSmoothnessLoss, self).__init__()
        self.func = Charbonnier(reduction=False)
        self.reduction = reduction
        self.getGrad = getGrad
        self.cs = cs

    def forward(self, input, img):
        idx  = torch.abs(img[:, :, :, 1:-1] - img[:, :, :, :-2]).mean(1, True) #[h, w-2]
        idx += torch.abs(img[:, :, :, 2:]  - img[:, :, :, 1:-1]).mean(1, True)

        idy  = torch.abs(img[:, :, 1:-1, :] - img[:, :, :-2, :]).mean(1, True) #[h-2, w]
        idy += torch.abs(img[:, :, 2:, :]  - img[:, :, 1:-1, :]).mean(1, True)

        dx = 2 * input[:, :, :, 1:-1] - input[:, :, :, :-2] - input[:, :, :, 2:]
        dy = 2 * input[:, :, 1:-1, :] - input[:, :, :-2, :] - input[:, :, 2:, :]
        wx = torch.exp(-self.cs * idx)
        wy = torch.exp(-self.cs * idy)
        loss_x = wx * self.func(dx)
        loss_y = wy * self.func(dy)

        if self.reduction:
            loss = loss_x.mean() + loss_y.mean()
        else:
            loss = loss_x[:, :, :-2, :] + loss_y[:, :, :, :-2] # [h-2, w-2]
        if self.getGrad:
            return loss, idx[:, :, :-2, :] + idy[:, :, :, :-2]

class TotalVariationLoss(nn.Module):
    def __init__(self, use_abs=True):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        N, C, H, W = img.shape
        count_h, count_w = (H - 1) * W, (W - 1) * H
        if use_abs:
            h_grad = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).sum()
            w_grad = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).sum()
        else:
            h_grad = torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2).sum()
            w_grad = torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2).sum()
        #return h_grad + w_grad
        loss = h_grad / count_h + w_grad / count_w
        return loss

def compute_tv_loss(img, use_abs=True):
    N, C, H, W = img.shape
    count_h, count_w = (H - 1) * W, (W - 1) * H
    if use_abs:
        h_grad = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).sum()
        w_grad = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).sum()
    else:
        h_grad = torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2).sum()
        w_grad = torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2).sum()
    #return h_grad + w_grad
    return h_grad / count_h + w_grad / count_w

class HDRCrit(object):
    def __init__(self, loss='l1', mu=5000):
        print('==> [%s]: %s' % (self.__class__.__name__, loss))
        if loss == 'l2':
            self.hdr_crit = torch.nn.MSELoss()
        elif loss == 'l1':
            self.hdr_crit = torch.nn.L1Loss()
        else:
            raise NotImplementedError('Loss function %s] is not implemented' % init_type)
        self.loss = loss
        self.mu = mu

    def __call__(self, pred, gt, weight=None, is_log=True):
        if not is_log:
            pred = eval_utils.pt_mulog_transform(pred, mu=self.mu)
            gt = eval_utils.pt_mulog_transform(gt, mu=self.mu)

        if weight is not None:
            loss = self.hdr_crit(pred * weight, gt * weight) / (weight.mean() + 1e-8)
        else:
            loss = self.hdr_crit(pred, gt)
        return loss

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        x_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().view(1, 1, 3, 3)
        y_filter = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().view(1, 1, 3, 3)
        self.register_buffer('x_filter', x_filter)
        self.register_buffer('y_filter', y_filter)

    def forward(self, input, target):
        input = input.mean(1, keepdim=True)
        target = target.mean(1, keepdim=True)

        gx_input = F.conv2d(input, self.x_filter)
        gx_target = F.conv2d(target, self.x_filter)
        gx_diff = torch.mean((gx_input - gx_target).pow(2))
        gy_input = F.conv2d(input, self.y_filter)
        gy_target = F.conv2d(target, self.y_filter)
        gy_diff = torch.mean((gy_input - gy_target).pow(2))
        diff = gx_diff + gy_diff
        # TODO: seprate compute
        with torch.no_grad():
            edge_input = torch.sqrt(torch.pow(gx_input, 2) + torch.pow(gy_input, 2))
            edge_target = torch.sqrt(torch.pow(gx_target, 2) + torch.pow(gy_target, 2))
        return diff, edge_input, edge_target

class EdgeLoss2(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss2, self).__init__()
        x_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().view(1, 1, 3, 3)
        y_filter = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().view(1, 1, 3, 3)
        self.register_buffer('x_filter', x_filter)
        self.register_buffer('y_filter', y_filter)

    def forward(self, input, target):
        input = input.mean(1, keepdim=True)
        target = target.mean(1, keepdim=True)
        gx_input = F.conv2d(input, self.x_filter)
        gy_input = F.conv2d(input, self.y_filter)
        gx_target = F.conv2d(target, self.x_filter)
        gy_target = F.conv2d(target, self.y_filter)
        # TODO: seprate compute
        edge_input = torch.sqrt(torch.pow(gx_input, 2) + torch.pow(gy_input, 2))
        edge_target = torch.sqrt(torch.pow(gx_target, 2) + torch.pow(gy_target, 2))
        diff = torch.mean((edge_input - edge_target).pow(2))
        return diff, edge_input, edge_target

# Feature space loss
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #for x in range(16, 23):
        #    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        #h = self.slice4(h)
        #h_relu4_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        out = OrderedDict({'relu1_2': h_relu1_2, 'relu2_2': h_relu2_2, 
                            'relu3_3': h_relu3_3}) #, 'relu4_3': h_relu4_3
        return out

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
class Vgg16Loss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3'], style_loss=True, style_w=120., requires_grad=False):
        super(Vgg16Loss, self).__init__()
        print('==> [%s]' % (self.__class__.__name__))
        self.style_loss = style_loss
        self.style_w = style_w

        self.vgg16 = Vgg16(requires_grad)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.layers = layers
        print("=> Parameters of VGG16 Losss: %d" % (mutils.get_params_num(self)))

    def forward(self, input, target):
        input  = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_feat = self.vgg16(input)
        target_feat = self.vgg16(target)
        loss = 0.0
        loss_term = OrderedDict()

        content_loss = 0
        for f_key in self.layers:
            content_loss += torch.nn.functional.l1_loss(input_feat[f_key], target_feat[f_key]) 
        loss_term['perc_loss'] = content_loss.item()
        loss += content_loss

        style_loss = 0
        if self.style_loss:
            for f_key in self.layers:
                style_loss += self.style_w * F.l1_loss(gram_matrix(input_feat[f_key]), gram_matrix(target_feat[f_key]))
            loss_term['style_f_loss'] = style_loss.item()
            loss += style_loss
        return loss, loss_term

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
