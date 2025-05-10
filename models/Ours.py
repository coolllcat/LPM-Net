import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
import time
from thop import profile

class Conv_layer(nn.Module):
    def __init__(self, c_in, c_out, use_relu=True, dense=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        self.acti = nn.LeakyReLU(inplace=True)
        self.use_relu = use_relu
        self.dense = dense

    def forward(self, x):
        if self.use_relu:
            y = self.acti(self.conv(x))
        else:
            y = self.conv(x)
        if self.dense:
            return torch.concat([y, x], dim=1)
        else:
            return y


class Dense_Module(nn.Module):
    def __init__(self, base_c, depth):
        super().__init__()

        self.layers = nn.ModuleList([Conv_layer(base_c*(i+1), base_c, dense=True) for i in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    

class Swin_Mamba_layer(nn.Module):
    def __init__(self, window_size, shift_size, dim, d_state = 8, d_conv = 4, expand = 1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand )
        
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size*window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = self.norm(x_windows)
        attn_windows = self.mamba(x_windows)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.permute(0, 3, 1, 2)

        return x
    

class Swin_Mamba_Module(nn.Module):
    def __init__(self, window_size, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([Swin_Mamba_layer(window_size = window_size, 
                                                      shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                                      dim = dim) 
                                     for i in range(depth)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class CfcCell(nn.Module):
    def __init__(self, hidden_size, out_size):
        super(CfcCell, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.ff1 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.ff2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.time_a = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.time_b = nn.Conv3d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.fc = nn.Conv3d(hidden_size, out_size, kernel_size=1)

    def forward(self, x):
        ff1 = self.ff1(x)
        ff2 = self.ff2(x)
        t_a = self.time_a(x)
        t_b = self.time_b(x)
        t_interp = self.sigmoid(t_a + t_b)
        out = ff1 * (1.0 - t_interp) + t_interp * ff2

        out = self.fc(out)
        ff1 = self.fc(ff1)
        ff2 = self.fc(ff2)
        return out, ff1, ff2


class PMLNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.base_c = args.base_c
        self.iscfc = args.iscfc
        self.depth_c = args.depth_c
        self.depth_m = args.depth_m
        self.fuse_method = args.fuse_method
        self.Conv_in = nn.Conv3d(2, self.base_c, kernel_size=3, stride=1, padding=1)
        self.Conv_P1 = Dense_Module(self.base_c, self.depth_c) if self.depth_c != 0 else nn.Identity()
        self.Conv_P2 = Dense_Module(self.base_c, self.depth_c) if self.depth_c != 0 else nn.Identity()
        self.Conv_P3 = Dense_Module(self.base_c, self.depth_c) if self.depth_c != 0 else nn.Identity()
        self.Mamba_P1 = Swin_Mamba_Module(window_size=16, dim=self.base_c*(self.depth_c+1), depth=self.depth_m) if self.depth_m != 0 else nn.Identity()
        self.Mamba_P2 = Swin_Mamba_Module(window_size=16, dim=self.base_c*(self.depth_c+1), depth=self.depth_m) if self.depth_m != 0 else nn.Identity()
        self.Mamba_P3 = Swin_Mamba_Module(window_size=16, dim=self.base_c*(self.depth_c+1), depth=self.depth_m) if self.depth_m != 0 else nn.Identity()
        self.cfc = CfcCell(self.base_c*(self.depth_c+1), 1) if self.iscfc else nn.Conv3d(self.base_c*(self.depth_c+1), 1, kernel_size=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.Conv_in(x)
        B, C, D, H, W = x.shape

        out_P1 = self.Conv_P1(x.permute(0,2,1,3,4).reshape(B*D, C, H, W))
        out_P2 = self.Conv_P2(x.permute(0,3,1,2,4).reshape(B*H, C, D, W))
        out_P3 = self.Conv_P3(x.permute(0,4,1,2,3).reshape(B*W, C, D, H))

        out_P1 = self.Mamba_P1(out_P1)
        out_P2 = self.Mamba_P2(out_P2)
        out_P3 = self.Mamba_P3(out_P3)

        C = out_P1.shape[1]
        out_P1 = out_P1.reshape(B, D, C, H, W).permute(0,2,1,3,4)
        out_P2 = out_P2.reshape(B, H, C, D, W).permute(0,2,3,1,4)
        out_P3 = out_P3.reshape(B, W, C, D, H).permute(0,2,3,4,1)

        if self.fuse_method == 'mean':
            out = (out_P1 + out_P2 + out_P3) / 3
        elif self.fuse_method == 'min':
            out = torch.min(torch.min(out_P1, out_P2), out_P3)
        elif self.fuse_method == 'max':
            out = torch.max(torch.max(out_P1, out_P2), out_P3)

        if self.iscfc:
            out, base, detail = self.cfc(out)
            out = F.tanh(out) / 2 + 0.5
            base = F.tanh(base) / 2 + 0.5
            detail = F.tanh(detail) / 2 + 0.5
        else:
            out = self.cfc(out)
            out = F.tanh(out) / 2 + 0.5
            base = None
            detail = None

        return out, base, detail


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--base_c', type=int, default=8)
    parser.add_argument('--depth_c', type=int, default=2)
    parser.add_argument('--depth_m', type=int, default=2)
    parser.add_argument('--iscfc', type=bool, default=True)
    parser.add_argument('--fuse_method', type=str, default='max')
    args = parser.parse_args() 

    # model = PMLNet(args)
    # mamba = Mamba(
    #         d_model=24,
    #         d_state=8, 
    #         d_conv=4, 
    #         expand=1 )
    # print(mamba)
    # print(mamba.A_log.shape)
    # print(model)
    # print(sum(p.numel() for p in model.parameters()))
    # print(sum(p.numel() for p in model.Conv_in.parameters()))
    # print(sum(p.numel() for p in model.Conv_P1.parameters()))
    # print(sum(p.numel() for p in model.Mamba_P1.parameters()))
    # print(sum(p.numel() for p in model.cfc.parameters()))
    # print(sum(p.numel() for p in mamba.parameters()))
    # print(sum(p.numel() for p in mamba.in_proj.parameters()))

    model = PMLNet(args).cuda()
    x = torch.rand(1,1,128,192,192).cuda()
    macs, params = profile(model, inputs=(x, x))
    tic = time.time()
    y = model(x, x)
    print(time.time()-tic)
    print(macs/1024/1024/1024)
    print(params/1024/1024)


