import torch
from torch import nn
from modules import ConvSC, Inception

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

# Encoder module
class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

# Decoder module
class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

import torch
from torch import nn
from modules import Inception

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)  # Element-wise multiplication for attention application

import torch
from torch import nn
from modules import Inception

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()
        self.N_T = N_T

        # Initialize encoder and decoder layers
        self.enc = nn.ModuleList([Inception(channel_in if i == 0 else channel_hid, channel_hid//2, channel_hid, incep_ker, groups) for i in range(N_T)])
        self.dec = nn.ModuleList([Inception(channel_hid if i < N_T-1 else 2*channel_hid, channel_hid//2, channel_hid if i < N_T-1 else channel_in, incep_ker, groups) for i in range(N_T)])

        # Attention module initialization
        self.attention = nn.ModuleList([nn.Conv2d(channel_hid, channel_hid, 1) for _ in range(N_T)])  # Using 1x1 conv to simulate attention

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # Flatten batch and time dimensions

        # Encoder with attention
        skips = []
        for i in range(self.N_T):
            x = self.enc[i](x)
            attention_out = self.attention[i](x)
            if i < self.N_T - 1:
                skips.append(attention_out)
            x = attention_out

        # Decoder
        for i in range(self.N_T):
            if i > 0:
                x = torch.cat([x, skips[-i]], dim=1)  # Concatenate skip connection
            x = self.dec[i](x)

        x = x.reshape(B, T, C, H, W)  # Restore original dimensions
        return x

# Ensure that attention modules do not alter the number of channels



# SimVP class for training
class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y
