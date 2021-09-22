from random import sample
import torch
from torch import nn
from torch.nn.modules.pixelshuffle import PixelShuffle

from model.STN import SpatialTransformer
from model.NLRN import ResidualBlcok



class NLST(nn.Module):
    def __init__(self, 
                 in_channels, 
                 work_channels, 
                 filter_size=64,
                 upsample=True,
                 iteration_steps=12, 
                 mode='embedded',
                 st_schedule="aaaahhhhtttt",
                 tps_grid_size=4,
                 **kwargs):
        """
        """
        super(NLST, self).__init__()
        assert len(st_schedule) == iteration_steps, \
            "Trnasofrm scheduel shoul have same length of iteration step number."      
        
        # params set
        self.steps = iteration_steps
        self.corr = None
        self.schedule = st_schedule
        self.upsample = upsample

        # FIXME 요거 데이터 맞춰야함.... RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)

        # modules set
        self.front = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, work_channels, 3, padding=1)
        )

        self.rb = ResidualBlcok(work_channels, mode=mode)
        if "a" in st_schedule:
            self.affine = SpatialTransformer(work_channels, filter_size, 3, 'affine')
        if "h" in st_schedule:
            self.homo= SpatialTransformer(work_channels, filter_size, 3, 'hom')
        if "t" in st_schedule:
            self.tps= SpatialTransformer(work_channels, filter_size, 3, 'tps', tps_grid_size)

        self.tail = nn.Sequential(
            nn.BatchNorm2d(work_channels),
            nn.ReLU(),
            nn.Conv2d(work_channels, 3, 3, padding=1)
        )

        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        if self.upsample:
            self.upsampler = nn.Sequential(
                nn.Conv2d(work_channels, 16 * work_channels, 3, padding=1),
                nn.PixelShuffle(4),
                nn.BatchNorm2d(work_channels),
                nn.ReLU(True),
                nn.Conv2d(work_channels, work_channels, 3, padding=1)
            )

    def forward(self, x):
        skip = x
        x = self.sub_mean(x)
        x = self.front(x)
        
        for tr in self.schedule:
            x = self.rb(x)
            if tr == "a":
                x, _ = self.affine(x)
            elif tr == "h":
                x, _ = self.homo(x)
            elif tr == "t":
                x, _ = self.tps(x)

        if self.upsample:
            x = self.upsampler(x)

        x = self.tail(x)
        x = self.add_mean(x)
        return x + skip


class MeanShift(nn.Conv2d):
    # TODO 이거는 데이터 mean 이랑 맞춰야함..
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


if __name__ == '__main__':
    # following code is for test
    import torch

    img = torch.zeros(2, 3, 64, 64)
    net = NLST(3, 128)
    out = net(img)
    print(out.size())