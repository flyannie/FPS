import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class ScaleEncoder(nn.Module): 
    def __init__(self, out_channel):
        super().__init__()
        self.main = nn.Sequential(
            BasicConv(4, out_channel//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel // 4, out_channel // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel // 2, out_channel, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channel, affine=True)
        )
    def forward(self, x):
        x = self.main(x)
        return x
    
class Fusion(nn.Module):
    def __init__(self, swin_channel, cnn_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(swin_channel, cnn_channel, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(swin_channel, cnn_channel, 3, stride=1, padding=1)

    def forward(self, swin_feature, cnn_feature):
        beta = self.conv1(swin_feature)
        gamma = self.conv2(swin_feature)
        output = cnn_feature * beta + gamma
        return output

class InputEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_resblock):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for i in range(num_resblock - 1):
            self.resblocks.append(ResBlock(in_channel, out_channel))
        self.resblocks.append(ResBlock(in_channel, out_channel, freq_filter=True))
        self.num_resblock = num_resblock
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.outconv = nn.Conv2d(in_channel*2, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        res = x.clone()
        for i, block in enumerate(self.resblocks):
            if i == self.num_resblock//4:
                skip = x
                x = self.down(x) 
            if i == self.num_resblock - self.num_resblock//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.outconv(torch.cat((x,skip), dim=1))
            x = block(x)
        return x + res
    
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, freq_filter=False):
        super().__init__()
        self.inconv = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.freq_filter = freq_filter
        self.freq_filter_k3 = Frequency_filter(out_channel//2, kernel_size=3) if freq_filter else nn.Identity()
        self.freq_filter_k5 = Frequency_filter(out_channel//2, kernel_size=5) if freq_filter else nn.Identity()
        self.global_ap = Global_AP(out_channel//2)
        self.local_ap = Local_AP(out_channel//2, patch_size=2)
        self.outconv = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        input_x = x
        x = self.inconv(x)
        if self.freq_filter:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out1 = self.freq_filter_k3(x1)
            out2 = self.freq_filter_k5(x2)
            x = torch.cat((out1, out2), dim=1)
        global_part, local_part = torch.chunk(x, 2,dim=1)
        global_part = self.global_ap(global_part)
        local_part = self.local_ap(local_part)
        out = torch.cat((global_part, local_part), dim=1)
        out = self.outconv(out) + input_x
        return out
    
class Frequency_filter(nn.Module):
    def __init__(self, in_channel, group=8, kernel_size=3, stride=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(in_channel, group*kernel_size**2, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.group = group
        self.kernel_size = kernel_size
        self.softmax = nn.Softmax(dim=-2)
        self.modulate = Modulate(in_channel)
    
    def forward(self, x):
        x_input = x 
        low_filter = self.gap(x) 
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2) 
        low_filter = self.softmax(low_filter) 
        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        high_part = x_input - low_part
        out = self.modulate(low_part, high_part)
        return out

class Modulate(nn.Module):
    def __init__(self, in_channel, r=2, base_channel=32, n=2):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        out_channel = max(int(in_channel / r), base_channel)
        self.fc = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.split = nn.ModuleList([])
        for i in range(n):
            self.split.append(nn.Conv2d(out_channel, in_channel, 1, 1, 0))
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(in_channel, in_channel, 1, 1, 0)

    def forward(self, low_part, high_part):
        sum_part = low_part + high_part 
        sum_part = self.gap(sum_part) 
        sum_part = self.fc(sum_part) 
        high_attention = self.split[0](sum_part)
        low_attention = self.split[1](sum_part)
        concate_attention = torch.cat([high_attention, low_attention], dim=1)
        concate_attention = self.softmax(concate_attention)
        high_attention, low_attention = torch.chunk(concate_attention, 2, dim=1)
        high_out = high_part * high_attention
        low_out = low_part * low_attention
        out = self.out(high_out + low_out)
        return out
    
class Global_AP(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.weight_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.weight_l = nn.Parameter(torch.zeros(in_channel), requires_grad=True)

    def forward(self, global_part):
        low_part = self.gap(global_part)
        high_part = (global_part - low_part) * (self.weight_h[None, :, None, None] + 1.)
        low_part = low_part * self.weight_l[None, :, None, None]
        return low_part + high_part

class Local_AP(nn.Module):
    def __init__(self, in_channel, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        channel = in_channel * patch_size **2
        self.weight_h = nn.Parameter(torch.zeros(channel), requires_grad=True)
        self.weight_l = nn.Parameter(torch.zeros(channel), requires_grad=True)

    def forward(self, local_part):
        patch = rearrange(local_part, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch = rearrange(patch, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)
        low_part = self.gap(patch)
        high_part = (patch - low_part) * self.weight_h[None, :, None, None]
        low_part = low_part * self.weight_l[None, :, None, None]
        out = high_part + low_part
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)
        return out
    
class MiddleEncoder(nn.Module):
    def __init__(self, in_channel, num_resblock):
        super().__init__()
        resblocks = [ResBlock(in_channel, in_channel) for _ in range(num_resblock - 1)]
        resblocks.append(ResBlock(in_channel, in_channel, freq_filter=True))
        self.resblocks = nn.Sequential(*resblocks)

    def forward(self, x):
        return self.resblocks(x)
    
class MiddleDecoder(nn.Module):
    def __init__(self, in_channel, num_resblock):
        super().__init__()
        layers = [ResBlock(in_channel, in_channel) for _ in range(num_resblock-1)]
        layers.append(ResBlock(in_channel, in_channel, freq_filter=True))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class OutputDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_resblock):
        super().__init__()
        self.resblocks = nn.ModuleList()
        for i in range(num_resblock - 1):
            self.resblocks.append(ResBlock(in_channel, out_channel))
        self.resblocks.append(ResBlock(in_channel, out_channel, freq_filter=True))
        self.num_resblock = num_resblock
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.outconv = nn.Conv2d(in_channel*2, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        res = x.clone()
        for i, block in enumerate(self.resblocks):
            if i == self.num_resblock//4:
                skip = x
                x = self.down(x)
            if i == self.num_resblock - self.num_resblock//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.outconv(torch.cat((x,skip), dim=1))
            x = block(x)
        return x + res
        





        
    