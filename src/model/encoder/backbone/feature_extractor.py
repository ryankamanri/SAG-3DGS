import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

class FeatureUNet(nn.Module):
    def __init__(self, output_scales=[1, 2, 4, 8, 16], out_channels=64):
        """
        输出多尺度特征图的UNet
        Args:
            output_scales: 需要输出的特征图尺度列表，每个元素表示下采样倍数
                           [1: 原图大小, 2: 1/2, 4: 1/4, 8: 1/8, ...]
        """
        super(FeatureUNet, self).__init__()
        
        self.output_scales = output_scales
        self.max_scale = max(output_scales) if output_scales else 1
        
        # 编码器
        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3, conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])
        
        # 解码器
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6], upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])
        
        # 特征融合层
        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(upconv_planes[6], upconv_planes[6])
        
        # 尺度特征输出层
        self.scale_outputs = nn.ModuleDict()
        for scale in output_scales:
            # 根据尺度选择对应的特征转换层
            if scale == 1:  # 原图大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[6], out_channels, 1)
            elif scale == 2:  # 1/2大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[5], out_channels, 1)
            elif scale == 4:  # 1/4大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[4], out_channels, 1)
            elif scale == 8:  # 1/8大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[3], out_channels, 1)
            elif scale == 16:  # 1/16大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[2], out_channels, 1)
            elif scale == 32:  # 1/32大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[1], out_channels, 1)
            elif scale == 64:  # 1/64大小
                self.scale_outputs[str(scale)] = nn.Conv2d(upconv_planes[0], out_channels, 1)
            else:
                raise ValueError(f"Unsupported output scale: {scale}. Valid scales: [1,2,4,8,16,32,64]")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # input [B, 3, H, W]
        # 存储各尺度特征图
        features = {}
        
        # 编码器路径
        out_conv1 = self.conv1(x)  # 1/2
        out_conv2 = self.conv2(out_conv1)  # 1/4
        out_conv3 = self.conv3(out_conv2)  # 1/8
        out_conv4 = self.conv4(out_conv3)  # 1/16
        out_conv5 = self.conv5(out_conv4)  # 1/32
        out_conv6 = self.conv6(out_conv5)  # 1/64
        out_conv7 = self.conv7(out_conv6)  # 1/128
        
        # 解码器路径
        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)  # 1/64
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        
        # 收集特征
        if 64 in self.output_scales:
            features[64] = self.scale_outputs["64"](out_iconv7)
        
        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)  # 1/32
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        
        if 32 in self.output_scales:
            features[32] = self.scale_outputs["32"](out_iconv6)
        
        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)  # 1/16
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        
        if 16 in self.output_scales:
            features[16] = self.scale_outputs["16"](out_iconv5)
        
        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)  # 1/8
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        
        if 8 in self.output_scales:
            features[8] = self.scale_outputs["8"](out_iconv4)
        
        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)  # 1/4
        concat3 = torch.cat((out_upconv3, out_conv2), 1)
        out_iconv3 = self.iconv3(concat3)
        
        if 4 in self.output_scales:
            features[4] = self.scale_outputs["4"](out_iconv3)
        
        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)  # 1/2
        concat2 = torch.cat((out_upconv2, out_conv1), 1)
        out_iconv2 = self.iconv2(concat2)
        
        if 2 in self.output_scales:
            features[2] = self.scale_outputs["2"](out_iconv2)
        
        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)  # 1/1
        out_iconv1 = self.iconv1(out_upconv1)
        
        if 1 in self.output_scales:
            features[1] = self.scale_outputs["1"](out_iconv1)
        
        # 按要求的尺度顺序返回特征图
        return [features[scale] for scale in self.output_scales if scale in features]