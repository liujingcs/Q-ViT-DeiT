"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import math

import torch
import torch.nn.functional as F

from ._quan_base import Qmodes
from .lsq_layer import grad_scale, round_pass, bit_pass, clamp, quantize_by_mse, QuantConv2d, QuantLinear

# import ipdb

__all__ = ['QuantWnConv2d', 'QuantWnLinear']


class QuantWnConv2d(QuantConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=-1,
                 mode=Qmodes.layer_wise, learned=True, mixpre=True):
        super(QuantWnConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode, learned=learned, mixpre=mixpre)
        self.is_second = False
        self.epsilon = None

    def initialize_scale(self, device):
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        # self.alpha.data.copy_(quantize_by_mse(self.weight))

        # normalize weight
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        weight = self.weight.add(-weight_mean).div(weight_std)

        quantize_by_mse(weight, self.alpha)
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # print(utils.get_rank(), self.alpha.data)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        n = int(nbits)
        # if self.init_state == 0:
        # print(f"initialize weight scale for int{self.nbits} quantization")
        # self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
        # self.alpha.data.copy_(quantize_by_mse(self.weight, Qn, Qp))
        # self.init_state.fill_(1)
        assert self.init_state == 1
        with torch.no_grad():
            g = 1.0 / math.sqrt(self.weight.numel() * Qp)
            # g = 1.0 / math.sqrt(self.weight.numel()) / Qp
        # g = 1.0 / math.sqrt(self.weight.numel()) / 4
        self.alpha.data.clamp_(min=1e-4)
        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha[n - 2], g)
        # w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = clamp(round_pass(self.weight / alpha), Qn, Qp) * alpha

        # normalize weight
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        weight = self.weight.add(-weight_mean).div(weight_std)

        self.x = weight
        if self.x.requires_grad:
            self.x.retain_grad()

        w_q = round_pass(clamp(weight / alpha, Qn, Qp)) * alpha

        if self.is_second:
            w_q = w_q + self.epsilon

        w_q = w_q * weight_std
        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def set_first_forward(self):
        self.is_second = False

    def set_second_forward(self):
        self.is_second = True

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("LSQ_wn_conv2d_qsamv2")
        return s


class QuantWnLinear(QuantLinear):
    def __init__(self, in_features, out_features, bias=True, nbits=-1, learned=True, mixpre=True, **kwargs):
        super(QuantWnLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits,
                                            learned=learned, mixpre=mixpre)
        self.is_second = False
        self.epsilon = None

    def initialize_scale(self, device):
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        # self.alpha.data.copy_(quantize_by_mse(self.weight))

        # normalize weight
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        weight = self.weight.add(-weight_mean).div(weight_std)

        quantize_by_mse(weight, self.alpha)
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        nbits = bit_pass(self.nbits)
        Qn = -2 ** (nbits - 1)
        Qp = 2 ** (nbits - 1) - 1
        n = int(nbits)
        # print(utils.get_rank(), self.alpha.data)
        # if self.init_state == 0:
        # self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
        # lsq+ init
        # m, v = self.weight.abs().mean(), self.weight.abs().std()
        # self.alpha.data.copy_(torch.max(torch.abs(m - 3*v), torch.abs(m + 3*v)) / 2 ** (self.nbits - 1) )
        assert self.init_state == 1
        with torch.no_grad():
            g = 1.0 / math.sqrt(self.weight.numel() * Qp)
            # g = 1.0 / math.sqrt(self.weight.numel()) / Qp
        # g = 1.0 / math.sqrt(self.weight.numel()) / 4
        self.alpha.data.clamp_(min=1e-4)
        # Method1:
        alpha = grad_scale(self.alpha[n - 2], g)
        # w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_q = clamp(round_pass(self.weight / alpha), Qn, Qp) * alpha

        # normalize weight
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        weight = self.weight.add(-weight_mean).div(weight_std)

        self.x = weight
        if self.x.requires_grad:
            self.x.retain_grad()

        w_q = round_pass(clamp(weight / alpha, Qn, Qp)) * alpha

        if self.is_second:
            w_q = w_q + self.epsilon

        w_q = w_q * weight_std

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)

    def set_first_forward(self):
        self.is_second = False

    def set_second_forward(self):
        self.is_second = True

    def extra_repr(self):
        s = super().extra_repr()
        s += ", method={}".format("LSQ_wn_linear_qsamv2")
        return s
