import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from model import common

#in-scale non-local attention
class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(NonLocalAttention, self).__init__()

        self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        
    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)

        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)

        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0,2,1).view(N,-1,H,W)


class StratumBlock(nn.Module):
    """ Repetative Attnetion module for each stratum """
    def __init__(self, channel, depth, concat=False):
        super().__init__()

        unit_stratum =  nn.Sequential(
                NonLocalAttention(channel, 1),
                nn.Conv2d(channel, channel, 3, 1, 1),
                nn.GELU()
                )
        self.stratum = nn.ModuleList([unit_stratum for _ in range(depth)])
        self.merge = nn.Sequential(
                nn.Conv2d(channel*depth, channel, 3, 1, 1),
                nn.GELU()
                )
        self.concat = concat

    def forward(self, x):
        res = x

        c = []
        for stra_unit in self.stratum:
            if self.concat:
                c.append(res)
            res = stra_unit(res)

        res = torch.cat(c, dim=1)
        res = self.merge(res)
        
        return x + res


class StrataAttentionModul(nn.Module):
    """ Residual Strata Block """
    def __init__(self, n_strata, in_dim=64, work_dim=64, reduction=[2, 4, 8, 16], concat=False):
        super().__init__()

        ch = work_dim

        self.strata_head = nn.ModuleList()
        self.strata_body = nn.ModuleList()
        self.strata_tail = nn.ModuleList()

        for i, r in enumerate(reduction[::-1]):
            # Create strata accoding to given channel and reduction list length
            self.strata_head.append(
                nn.Sequential(nn.Conv2d(in_dim, ch, 1, padding=0, bias=True), 
                              nn.GELU(),
                              *[nn.Conv2d(ch, ch, 6, 2, 2), nn.GELU()] * int(math.log2(r))
                              )
                )
            self.strata_body.append(StratumBlock(ch, n_strata, concat))
            
            self.strata_tail.append(
                nn.Sequential(*[nn.ConvTranspose2d(ch, ch, 6, 2, 2), nn.GELU()] * int(math.log2(r)))
                )
        
        self.merge_conv = nn.Sequential(
            nn.Conv2d(ch*len(reduction), in_dim, 3, 1, 1),
            nn.GELU()
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : B X C X H X W
        """
        outs = []
        for head, body, tail in zip(self.strata_head, self.strata_body, self.strata_tail):
            out = tail(body(head(x)))
            outs.append(out)

        res = torch.cat(outs, dim=1)
        res = self.merge_conv(res)

        return x + res


## Non-local Strata Transformer Network (NLSTN)
class NLST(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLST, self).__init__()
        
        self.n_strablocks = args.n_strablocks
        n_stratum = args.n_stratum
        n_feats = args.n_feats
        kernel_size = 3
        work_dim = args.work_dim
        reduction = args.reduction 
        scale = args.scale
        concat = args.concat
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
           StrataAttentionModul(n_stratum, n_feats, work_dim, reduction, concat) \
            for _ in range(self.n_strablocks)]

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