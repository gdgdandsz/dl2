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


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()
        self.N_T = N_T
        self.spatial_attention = SpatialAttentionModule()
        self.temporal_attention = TemporalAttentionModule(channel_hid)

        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # Encoder with spatial attention
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            z = self.spatial_attention(z)  # Apply spatial attention
            if i < self.N_T - 1:
                skips.append(z)

        # Apply temporal attention before decoding
        z = z.reshape(B, T, C, H, W)
        z = self.temporal_attention(z)
        z = z.reshape(B, T*C, H, W)

        # Decoder
        for i in range(self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1)) if i > 0 else self.dec[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.compress = nn.Conv2d(2, 1, 1, bias=False)  # 压缩输入的两个通道到一个通道
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用全局平均池化和最大池化来获取通道的注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.compress(attn)
        attn = self.sigmoid(attn)
        return x * attn


class TemporalAttentionModule(nn.Module):
    def __init__(self, channels, heads=8):
        super(TemporalAttentionModule, self).__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x should be reshaped to (Batch, Channel, Time) before being passed here
        B, C, T = x.size()
        q = self.query(x).view(B, self.heads, C // self.heads, T)
        k = self.key(x).view(B, self.heads, C // self.heads, T)
        v = self.value(x).view(B, self.heads, C // self.heads, T)

        q = q.permute(0, 1, 3, 2)  # Prepare for batch matrix multiplication
        k = k.permute(0, 1, 3, 2)
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale)
        x = torch.matmul(attn, v).view(B, C, T)
        return x


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
