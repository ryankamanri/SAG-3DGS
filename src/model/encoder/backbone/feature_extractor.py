import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FeatureNet(nn.Module):
    def __init__(self, norm_act=nn.BatchNorm2d, stage_channels=[12, 24, 48]):
        super(FeatureNet, self).__init__()
        self.in_channel = 3
        self.stage_channels = stage_channels
        self.conv0 = nn.Sequential(
                        ConvBnReLU(self.in_channel, self.stage_channels[0], 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(self.stage_channels[0], self.stage_channels[0], 3, 1, 1, norm_act=norm_act))
        self.conv1 = nn.Sequential(
                        ConvBnReLU(self.stage_channels[0], self.stage_channels[1], 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(self.stage_channels[1], self.stage_channels[1], 3, 1, 1, norm_act=norm_act))
        self.conv2 = nn.Sequential(
                        ConvBnReLU(self.stage_channels[1], self.stage_channels[2], 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(self.stage_channels[2], self.stage_channels[2], 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(self.stage_channels[2], self.stage_channels[2], 1)
        self.lat1 = nn.Conv2d(self.stage_channels[1], self.stage_channels[2], 1)
        self.lat0 = nn.Conv2d(self.stage_channels[0], self.stage_channels[2], 1)

        self.smooth1 = nn.Conv2d(self.stage_channels[2], self.stage_channels[1], 3, padding=1)
        self.smooth0 = nn.Conv2d(self.stage_channels[2], self.stage_channels[0], 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # x: Tensor(B, V, C, H, W)
        b, v, c, h, w = x.shape
        x = x.view(b*v, c, h, w)
        
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        # feat1 = self.smooth1(feat1)
        # feat0 = self.smooth0(feat0)
        return feat0.view(b, v, -1, h, w)

    
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=192):
        super(CNNFeatureExtractor, self).__init__()
        
        cur_channels = 16
        self.conv_in_channels_list = [in_channels] # [3]
        self.conv_out_channels_list = [cur_channels] # [8]
        while cur_channels * 2 < out_channels:
            self.conv_in_channels_list.append(cur_channels) # [3, 8, 16, 32, 64]
            self.conv_out_channels_list.append(cur_channels * 2) # [8, 16, 32, 64, 128]
            cur_channels *= 2
        self.conv_in_channels_list.append(cur_channels) # [3, 8, 16, 32, 64, 128]
        self.conv_out_channels_list.append(out_channels) # [8, 16, 32, 64, 128, 192]
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1) 
            for in_c, out_c in zip(self.conv_in_channels_list, self.conv_out_channels_list)])
        
    
    def forward(self, x):
        # x: Tensor(B, V, C, H, W)
        b, v, c, h, w = x.shape
        x = x.view(b*v, c, h, w)
        for conv in self.convs:
            x = conv(x)

        return x.view(b, v, -1, h, w)