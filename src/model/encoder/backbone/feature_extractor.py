import torch
import torch.nn as nn
import torch.nn.functional as F


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, 
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class MultiViewFeatureExtractor(nn.Module):
    """
    Modified from `FeatureNet` in `src/model/encoder/mvsnet/cas_module.py`
    input: 
        x: Tensor(B, V, C, H, W)
        
    output: 3 stage features.
    """
    def __init__(self, base_channels=8, num_stage=3):
        super(MultiViewFeatureExtractor, self).__init__()
        assert num_stage == 3
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        
        self.fuse1 = nn.Conv2d(base_channels * 2, base_channels * 4, 1, bias=False)
        self.fuse0 = nn.Conv2d(base_channels, base_channels * 4, 1, bias=False)

        self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 4, 3)
        self.deconv2 = DeConv2dFuse(base_channels * 4, base_channels * 4, 3)

        self.out2 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out3 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        

    def forward(self, x: torch.Tensor):
        b, v, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        out = out.view(b, v, -1, h // 4, w // 4)
        outputs.append(out)

        intra_feat = self.deconv1(self.fuse1(conv1), intra_feat)
        out = self.out2(intra_feat)
        out = out.view(b, v, -1, h // 2, w // 2)
        outputs.append(out)

        intra_feat = self.deconv2(self.fuse0(conv0), intra_feat)
        out = self.out3(intra_feat)
        out = out.view(b, v, -1, h, w)
        outputs.append(out)

        return outputs
    
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=192):
        super(CNNFeatureExtractor, self).__init__()
        
        cur_channels = 16
        self.conv_in_channels_list = [in_channels] # [3]
        self.conv_out_channels_list = [cur_channels] # [16]
        while cur_channels * 2 < out_channels:
            self.conv_in_channels_list.append(cur_channels) # [3, 16, 32, 64]
            self.conv_out_channels_list.append(cur_channels * 2) # [16, 32, 64, 128]
            cur_channels *= 2
        self.conv_in_channels_list.append(cur_channels) # [3, 16, 32, 64, 128]
        self.conv_out_channels_list.append(out_channels) # [16, 32, 64, 128, 192]
        
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