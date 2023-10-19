from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear")

class ConvBlock(nn.Module):
    def __init__(self, num_ch_in, num_ch_out, is_bn=False, is_act=True):
        super().__init__()
        self.conv = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.is_bn = is_bn
        self.is_act = is_act
        if self.is_bn:
            self.bn = nn.BatchNorm2d(num_ch_out)
        if self.is_act:
            self.act = nn.ELU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        if self.is_bn:
            out = self.bn(out)
        if self.is_act:
            out = self.act(out)
        return out

class DecoderMainBlock(nn.Module):
    def __init__(self, num_ch_in, num_ch_out):
        super().__init__()
        self.conv_0 = ConvBlock(num_ch_in, num_ch_out, is_bn=False, is_act=True)
        self.conv_1 = ConvBlock(num_ch_out, num_ch_out, is_bn=False, is_act=True)

    def forward(self, x):
        out = torch.cat(x, dim=1)
        out = self.conv_0(out)
        out = upsample(out)
        out = self.conv_1(out)
        return out

class Decoder1LastBlock(nn.Module):
    def __init__(self, num_ch_in, num_ch_out):
        super().__init__()
        self.conv = ConvBlock(num_ch_in, num_ch_out, is_bn=False, is_act=True)

    def forward(self, x):
        out = torch.cat(x, dim=1)
        out = self.conv(out)
        return out
        
class DecoderFirstBlock(nn.Module):
    def __init__(self, num_ch_in, num_ch_out):
        super().__init__()
        self.conv = ConvBlock(num_ch_in, num_ch_out, is_bn=False, is_act=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = upsample(out)
        return out
       
class DispBlock(nn.Module):
    def __init__(self, num_ch_in):
        super().__init__()
        self.conv = ConvBlock(num_ch_in, 1, is_bn=False, is_act=False)
        self.sigmoid = nn.Sigmoid()
      
    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, enc_name):
        super().__init__()
        
        if enc_name == "SwiftFormer_XS":
            self.num_ch_enc = [24] + num_ch_enc 
            self.num_ch_dec_1 = np.array([12, 24, 28, 56, 110]) 
            self.num_ch_dec_2 = np.array([24, 24, 28, 55])
        elif enc_name == "SwiftFormer_S":
            self.num_ch_enc = [24] + num_ch_enc
            self.num_ch_dec_1 = np.array([12, 24, 32, 84, 112]) 
            self.num_ch_dec_2 = np.array([24, 24, 42, 56]) 
        else:
            raise TypeError("wrong config file")

        self.num_dec_1_stages = len(self.num_ch_dec_1)
        self.num_dec_2_stages = len(self.num_ch_dec_2)

        self.num_out_scales = 3

        self.network = nn.ModuleDict()

        for i in range(self.num_dec_1_stages):
            if i == self.num_dec_1_stages - 1:
                self.network[f"dec_1_{i}"] = DecoderFirstBlock(self.num_ch_enc[i], self.num_ch_dec_1[i])
            elif i > 0:
                self.network[f"dec_1_{i}"] = DecoderMainBlock(self.num_ch_dec_1[i + 1] + self.num_ch_enc[i], self.num_ch_dec_1[i])
            else:
                self.network[f"dec_1_{i}"] = Decoder1LastBlock(self.num_ch_dec_1[i + 1] + self.num_ch_enc[i], self.num_ch_dec_1[i])
            
        for i in range(self.num_dec_2_stages):
            if i == self.num_dec_2_stages - 1:
                self.network[f"dec_2_{i}"] = DecoderFirstBlock(self.num_ch_dec_1[i + 1], self.num_ch_dec_2[i])
            elif i > 0:
                self.network[f"dec_2_{i}"] = DecoderMainBlock(self.num_ch_dec_2[i + 1] + self.num_ch_dec_1[i + 1], self.num_ch_dec_2[i])
            else:
                self.network[f"dec_2_{i}"] = DecoderMainBlock(self.num_ch_dec_2[i + 1] + self.num_ch_dec_1[i + 1] + self.num_ch_dec_1[i], self.num_ch_dec_2[i])
        
        for i in range(self.num_out_scales):
            self.network[f"disp_{i}"] = DispBlock(self.num_ch_dec_2[i])

    def forward(self, in_features):

        disps = {}
        outs = {}

        for i in range(self.num_dec_1_stages - 1, -1, -1):
            if i == self.num_dec_1_stages - 1:
                x = self.network[f"dec_1_{i}"](in_features[i]) 
            else:
                x = self.network[f"dec_1_{i}"]([x, in_features[i]]) 
            outs[f"dec_1_out_{i}"] = x  

        for i in range(self.num_dec_2_stages - 1, -1, -1):
            if i == self.num_dec_2_stages - 1:
                x = self.network[f"dec_2_{i}"](outs[f"dec_1_out_{i + 1}"]) 
            elif i > 0:
                x = self.network[f"dec_2_{i}"]([x, outs[f"dec_1_out_{i + 1}"]])
            else:
                x = self.network[f"dec_2_{i}"]([x, outs[f"dec_1_out_{i + 1}"], outs[f"dec_1_out_{i}"]])
            outs[f"dec_2_out_{i}"] = x  

        for i in range(self.num_out_scales):
            disps[("disp", i)] = self.network[f"disp_{i}"](outs[f"dec_2_out_{i}"])

        return disps