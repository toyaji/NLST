import torch
from torch import nn

from model.STN import SpatialTransformer
from model.NLRN import ResidualBlcok


class NLST(nn.Module):
    def __init__(self, 
                 in_channels, 
                 work_channels, 
                 filter_size=64, 
                 steps=12, 
                 mode='embedded',
                 st_schedule="aaaahhhhtttt",
                 tps_grid_size=4):
        """
        """
        super(NLST, self).__init__()
        assert len(st_schedule) == steps, \
            "Trnasofrm scheduel shoul have same length of iteration step number."      
        
        # params set
        self.steps = steps
        self.corr = None
        self.schedule = st_schedule

        # TODO 앞단에 cnn_geo_transform 붙일지 확인필요함
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
            nn.Conv2d(work_channels, 1, 3, padding=1)
        )

    def forward(self, x):
        skip = x
        x = self.front(x)
        
        for tr in self.schedule:
            x, self.corr = self.rb(x, self.corr)
            if tr == "a":
                x, _ = self.affine(x)
            elif tr == "h":
                x, _ = self.homo(x)
            elif tr == "t":
                x, _ = self.tps(x)

        x = self.tail(x)
        return x + skip



if __name__ == '__main__':
    # following code is for test
    import torch

    img = torch.zeros(2, 3, 64, 64).cuda()
    net = NLST(3, 128).cuda()
    out = net(img)
    print(out.size())