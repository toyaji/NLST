import torch
from torch import nn
from torch.nn import functional as F


class NLRN(nn.Module):
    def __init__(self, in_channels, work_channels, steps, mode) -> None:
        super(NLRN, self).__init__()
        
        self.front = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, work_channels, 3, padding=1)
        )

        # iterative residual block, here we'll put 12 steps basically.
        self.rb = ResidualBlcok(work_channels, mode=mode)

        self.tail = nn.Sequential(
            nn.BatchNorm2d(work_channels),
            nn.ReLU(),
            nn.Conv2d(work_channels, 1, 3, padding=1)
        )

        self.steps = steps
        self.corr = None

    def forward(self, x):
        skip = x
        x = self.front(x)
        
        for _ in range(self.steps):
            x, self.corr = self.rb(x, self.corr)

        x = self.tail(x)
        return x + skip


class ResidualBlcok(nn.Module):
    def __init__(self, in_channels, mode) -> None:
        super(ResidualBlcok, self).__init__()

        self.front = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU())

        self.body = NLBlockND(in_channels, mode=mode)
                
        self.tail = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
                )
    
    def forward(self, x, corr):
        skip = x
        x = self.front(x)
        x, corr = self.body(x, corr)
        x = self.tail(x)
        x += skip
        return x, corr


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block 
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    nn.BatchNorm2d(self.in_channels)
                )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x, corr):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        # we need to iteratively flow the correlation
        if corr is not None:
            f += corr

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z, f



if __name__ == '__main__':
    # following code is for test
    import torch

    for bn_layer in [True, False]:

        img = torch.zeros(2, 3, 64, 64).cuda()
        net = NLRN(in_channels=3, work_channels=128, steps=5, mode='embedded').cuda()
        out = net(img)
        print(out.size())