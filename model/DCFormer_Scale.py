import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp

from timm.models.layers import DropPath


class Model(nn.Module):
    def __init__(self, args=None, in_dim=2, out_dim=2):
        super().__init__()

        if args == None:
            layers, d_hid, frames=6, 256, 27
            num_joints_in, num_joints_out = 17, 17
            MAE = False
        else:
            layers, d_hid, frames = args.layers, args.d_hid, args.frames
            num_joints_in, num_joints_out = args.n_joints, args.out_joints
            MAE = args.MAE

        self.pose_emb = nn.Linear(in_dim, d_hid, bias=False)
        self.gelu = nn.GELU()
        self.staformer = STAFormer(layers, frames, num_joints_in, d_hid)
        self.regress_head = nn.Linear(d_hid, out_dim, bias=False)

        #masks
        if MAE:
            self.dec_pos_embedding = nn.Parameter(torch.randn(1, frames, num_joints_in, in_dim))
            self.mask_token = nn.Parameter(torch.randn(1, 1, num_joints_in, in_dim))
            self.spatial_mask_token = nn.Parameter(torch.randn(1, 1, 2))       
            self.spatial_mask_num = args.spatial_mask_num


    def forward(self, x, pre_mask=False, mask=None, spatial_mask=None, is_3dhp=False):
        # b, t, s, c = x.shape  #batch,frame,joint,coordinate
        # dimension tranfer
        if is_3dhp:
            x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()  # B,T,J,2,1 (for 3dhp)

        if pre_mask:
            b, f, s, c = x.shape  #batch,frame,joint,coordinate
            x_spatial_mask = x.clone()
            x_spatial_mask[:,spatial_mask] = self.spatial_mask_token.expand(b,f*self.spatial_mask_num,2)
            x = x_spatial_mask

        x = self.pose_emb(x)
        x = self.gelu(x)
        # spatio-temporal correlation
        x = self.staformer(x)
        # regression head
        x = self.regress_head(x)

        return x


class STAF_ATTENTION(nn.Module):
    def __init__(self, d_time, d_joint, d_coor, head=8):
        super().__init__()
        # print(d_time, d_joint, d_coor,head)
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)

        self.scale = (d_coor // 2) ** -0.5
        self.proj = nn.Linear(d_coor, d_coor) 
        self.proj_s = nn.Linear(d_coor//2, d_coor) #
        self.proj_t = nn.Linear(d_coor//2, d_coor) #
        self.d_time = d_time
        self.d_joint = d_joint
        self.head = head

        # LDPE
        self.ldpe_t = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.ldpe_s  = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        #fusion
        self.fusion = nn.Linear(d_coor , 2)
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5) 

        # HDPE
        self.hdpe_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
        self.hdpe_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)

        self.drop = DropPath(0.5)

    def forward(self, input):
        b, t, s, c = input.shape

        h = input
        x = self.layer_norm(input)

        qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c
        qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c

        # space group and time group
        qkv_s, qkv_t = qkv.chunk(2, 4)  # [3,b,t,s,c//2],  [3,b,t,s,c//2]

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c//2
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c//2

        # reshape for mat
        q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)  # b,t,s,c//2-> b*h*t,s,c//2//h
        k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)  # b,t,s,c//2-> b*h*t,c//2//h,s

        q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)  # b,t,s,c//2 -> b*h*s,t,c//2//h
        k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  # b,t,s,c//2->  b*h*s,c//2//h,t

        att_s = (q_s @ k_s) * self.scale  # b*h*t,s,s
        att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t

        att_s = att_s.softmax(-1)  # b*h*t,s,s
        att_t = att_t.softmax(-1)  # b*h*s,t,t

        v_s = rearrange(v_s, 'b  t s c -> b c t s ')
        v_t = rearrange(v_t, 'b  t s c -> b c t s ')

        # HLDPE 
        hdpe_s = self.hdpe_s(v_s)  # b,c//2,t,s
        hdpe_t = self.hdpe_t(v_t)  # b,c//2,t,s
        ldpe_s = self.ldpe_s(hdpe_s)
        ldpe_t = self.ldpe_t(hdpe_t)
        ldpe_s = rearrange(ldpe_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        ldpe_t = rearrange(ldpe_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h
        hdpe_s = rearrange(hdpe_s, 'b (h c) t s  -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        hdpe_t = rearrange(hdpe_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        # MSA
        v_s = rearrange(v_s, 'b (h c) t s   -> (b h t) s c ', h=self.head)  # b*h*t,s,c//2//h
        v_t = rearrange(v_t, 'b (h c) t s  -> (b h s) t c ', h=self.head)  # b*h*s,t,c//2//h

        #print(self.sep_weight)
        x_s = att_s @ v_s + hdpe_s + 1e-4 * self.drop(ldpe_s)               # b*h*t,s,c//2//h 
        x_t = att_t @ v_t + hdpe_t  + 1e-9 * self.drop(ldpe_t)               # b*h*s,t,c//2//h

        x_s = rearrange(x_s, '(b h t) s c -> b h t s c ', h=self.head, t=t)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2 
        x_t = rearrange(x_t, '(b h s) t c -> b h t s c ', h=self.head, s=s)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2 

        x_s = rearrange(x_s, 'b h t s c -> b  t s (h c) ')  # b,t,s,c//2
        x_t = rearrange(x_t, 'b h t s c -> b  t s (h c) ')  # b,t,s,c//2
        x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
        alpha = self.fusion(x).softmax(dim=-1)
        x = self.proj_s(x_s) * alpha[..., 0:1] + self.proj_t(x_t) * alpha[..., 1:2]

        # projection and skip-connection
        x = self.proj(x)
        x = x + h
        return x


class STAF_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)

        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)

        self.staf_att = STAF_ATTENTION(d_time, d_joint, d_coor)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.staf_att(input)
        x = x + self.drop(self.mlp(self.layer_norm(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class STAFormer(nn.Module):
    def __init__(self, num_block, d_time, d_joint, d_coor ):
        super(STAFormer, self).__init__()

        self.num_block = num_block
        self.d_time = d_time
        self.d_joint = d_joint
        self.d_coor = d_coor

        self.staf_block = []
        for l in range(self.num_block):
            self.staf_block.append(STAF_BLOCK(self.d_time, self.d_joint, self.d_coor))
        self.staf_block = nn.ModuleList(self.staf_block)

    def forward(self, input):
        # blocks layers
        for i in range(self.num_block):
            input = self.staf_block[i](input)
        # exit()
        return input



if __name__ == "__main__":
    # inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = Model(out_dim=2)
    inputs = torch.rand([1, 243, 17, 2])
    output = net(inputs)
    print(output.size())
    from thop import profile

    flops, params = profile(net, inputs=(inputs,))
    print(flops)
    print(params)
