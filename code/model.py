import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.SwinTransformer import SwinTransformer
from modules.BasicBlock import ScaleEncoder, Fusion, BasicConv, InputEncoder, MiddleEncoder, MiddleDecoder, OutputDecoder

class HFSNet(nn.Module):
    def __init__(self, base_channel=32, num_resblock=8):
        super(HFSNet,self).__init__()
        self.SwinTransformer = SwinTransformer()
        self.ScaleEncoder_2 = ScaleEncoder(base_channel * 2)
        self.ScaleEncoder_4 = ScaleEncoder(base_channel * 4)
        self.ScaleEncoder_8 = ScaleEncoder(base_channel * 8)
        self.Fusion_2 = Fusion(96, 64)
        self.Fusion_4 = Fusion(192, 128)
        self.Fusion_8 = Fusion(384, 256)
        self.FeatureExtract = nn.ModuleList([
            BasicConv(4, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*8, kernel_size=3, relu=True, stride=2),
            
            BasicConv(base_channel*8, base_channel*4, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 2, kernel_size=3, relu=False, stride=1)
        ])
        self.Encoder_1 = InputEncoder(base_channel, base_channel, num_resblock)
        self.Fusion_Block_1 = BasicConv(base_channel*4, base_channel*2, kernel_size=3, stride=1, relu=False)
        self.Encoder_2 = MiddleEncoder(base_channel*2, num_resblock)
        self.Fusion_Block_2 = BasicConv(base_channel*8, base_channel*4, kernel_size=3, stride=1, relu=False)
        self.Encoder_3 = MiddleEncoder(base_channel*4, num_resblock)
        self.Fusion_Block_3 = BasicConv(base_channel*16, base_channel*8, kernel_size=3, stride=1, relu=False)
        self.Encoder_4 = MiddleEncoder(base_channel*8, num_resblock)
        
        self.Decoder_1 = MiddleDecoder(base_channel*8, num_resblock)
        self.OutConv_8 = nn.ModuleList([
            BasicConv(base_channel*8, base_channel*4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel*4, 2, kernel_size=3, relu=False, stride=1)
        ])
        self.DeConv_1 = BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1)
        self.Decoder_2 = MiddleDecoder(base_channel*4, num_resblock)
        self.OutConv_4 = BasicConv(base_channel * 4, 2, kernel_size=3, relu=False, stride=1)
        self.DeConv_2 = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.Decoder_3 = MiddleDecoder(base_channel*2, num_resblock)
        self.OutConv_2 = BasicConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1)
        self.DeConv_3 = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        self.Decoder_4 = OutputDecoder(base_channel, base_channel, num_resblock)

    def forward(self, x):
        outs = list()
        SwinTouts = self.SwinTransformer(x)      
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_8 = F.interpolate(x_4, scale_factor=0.5)

        scale_feature_2 = self.ScaleEncoder_2(x_2)
        scale_feature_4 = self.ScaleEncoder_4(x_4)
        scale_feature_8 = self.ScaleEncoder_8(x_8)

        x1 = self.FeatureExtract[0](x)
        skip1 = self.Encoder_1(x1)
        M1 = self.FeatureExtract[1](skip1)
        fusion_1 = self.Fusion_2(SwinTouts[0], M1)
        MI1=torch.cat([scale_feature_2, fusion_1], dim=1)

        x2 = self.Fusion_Block_1(MI1)
        skip2 = self.Encoder_2(x2)
        M2 = self.FeatureExtract[2](skip2)
        fusion_2 = self.Fusion_4(SwinTouts[1], M2)
        MI2 = torch.cat([scale_feature_4, fusion_2], dim=1)

        x3 = self.Fusion_Block_2(MI2)
        skip3 = self.Encoder_3(x3)
        M3 = self.FeatureExtract[3](skip3)
        fusion_3 = self.Fusion_8(SwinTouts[2], M3)
        MI3 = torch.cat([scale_feature_8, fusion_3], dim=1)

        x4 = self.Fusion_Block_3(MI3)
        x4 = self.Encoder_4(x4)

        x5 = self.Decoder_1(x4)
        out_8 = self.OutConv_8[0](x5) 
        out_8 = self.OutConv_8[1](out_8)
        outs.append(out_8)
        x5 = self.FeatureExtract[4](x5)
        x5 = torch.cat([x5, skip3], dim=1)
        x5 = self.DeConv_1(x5)

        x6 = self.Decoder_2(x5)
        out_4 = self.OutConv_4(x6)
        outs.append(out_4)
        x6 = self.FeatureExtract[5](x6)
        x6 = torch.cat([x6, skip2], dim=1)
        x6 = self.DeConv_2(x6)

        x7 = self.Decoder_3(x6)
        out_2 = self.OutConv_2(x7)
        outs.append(out_2)
        x7 = self.FeatureExtract[6](x7)
        x7 = torch.cat([x7, skip1], dim=1)
        x7 = self.DeConv_3(x7)

        x8 = self.Decoder_4(x7)
        out = self.FeatureExtract[7](x8)
        outs.append(out)

        return outs



        
        







        
