import torch
import torch.nn as nn
import torchvision.models.vgg as vgg

from collections import namedtuple

from model import common

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X 64 X H X W)
            returns :
                out : list of feature maps(B X 128 X H/2 X W/2, 
                                           B X 256 X H/4 X W/4,
                                           B X 512 X H/8 X W/8,
                                           B X 512 X H/16 X W/16)
        """
        h = self.slice2(x)   
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out

class StackChannelAttention(nn.Module):
    def __init__(self, extractor, in_dim=64, channel=[128, 256, 512, 512], reduction=[2, 4, 8, 8]):
        super().__init__()

        self.vgg = extractor
        self.avg_pool = nn.ModuleList([nn.AdaptiveAvgPool2d(1)] * 4)
        self.conv_sqeeze = nn.ModuleList()

        for ch, r in zip(channel, reduction):
            self.conv_sqeeze.append(nn.Sequential(
                nn.Conv2d(ch, ch // r, 1, padding=0, bias=True),
                nn.ReLU(inplace=True))
            )

        self.merge_conv = nn.Conv2d(in_dim*4, in_dim, 1, padding=0, bias=True)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : B X C X H X W
        """
        outs = self.vgg(x)
        pooled = []
        for pool, sqeeze, out in zip(self.avg_pool, self.conv_sqeeze, outs):
            pooled.append(pool(sqeeze(out)))
            
        pooled = torch.cat(pooled, dim=1)
        attn = self.merge_conv(pooled)

        return x * attn

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, extractor, n_feat, kernel_size, channels, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(
            StackChannelAttention(extractor, n_feat, channels, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, extractor, n_feat, kernel_size, channels, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()

        #extractor = VGG19(requires_grad=True)

        modules_body = []
        modules_body = [
            RCAB(
                conv, extractor, n_feat, kernel_size, channels, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Stack Channel Attention Network (RCAN)
class SCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        channels = args.channels
        reduction = args.reduction 
        scale = args.scale
        
        extractor = VGG19(requires_grad=True)
        act = nn.GELU()
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, extractor, n_feats, kernel_size, channels, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x