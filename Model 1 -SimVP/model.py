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

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, H, W, incep_ker, groups):
        super(Mid_Xnet, self).__init__()
        self.N_T = N_T
        # 计算输入到多头注意力机制的维度应为 H*W，我们使用这个维度进行下采样
        self.downscale = nn.Linear(H * W, 128)  # 从H*W降到128
        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

        # 初始化Inception层
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker, groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker, groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker, groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker, groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker, groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker, groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, T, C * H * W)  # 将所有维度合并

        # 将特征从每个通道的空间尺寸降维
        x = x.view(B * T * C, H * W)
        x = self.downscale(x)

        # 调整形状以符合多头注意力的要求
        x = x.view(B, T * C, 128)

        # 应用Multihead Attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # 调整输出形状回原始维度，这里需要根据实际情况修改
        x = attn_output.view(B, T, C, H, W)
        x = x.reshape(B, T * C, H, W)

        # 通过Inception模块进行编码和解码
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.view(B, T, C, H, W)
        return y


# SimVP class for training
class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, H, W, incep_ker, groups)
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
