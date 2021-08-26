# This code come form https://github.com/aicaffeinelife/Pytorch-STN/tree/master/models


""" A plug and play Spatial Transformer Module in Pytorch """ 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from model.transformation import AffineGridGen, HomographyGridGen, TpsGridGen

# for test import
#from transformation import AffineGridGen, HomographyGridGen, TpsGridGen


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.
    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self,
                 in_channels, 
                 in_channel_size, 
                 kernel_size, 
                 geo_model='affine',
                 tps_grid_size=3,
                 tps_reg_factor=0, 
                 use_dropout=False):
                 
        super(SpatialTransformer, self).__init__()

        if isinstance(in_channel_size, int):
            self._h, self._w = in_channel_size, in_channel_size
        elif isinstance(in_channel_size, tuple):
            self._h, self._w = in_channel_size 

        self.geo_model = geo_model
        self._in_ch = in_channels 
        self._ksize = kernel_size
        self.dropout = use_dropout
        self.pooled_size = int(self._h * self._w / 64)

        # affine grid according to the given model
        if self.geo_model=='affine':
            self.params = 6
            self.gridGen = AffineGridGen(self._h, self._w, self._in_ch)
        elif self.geo_model=='hom':
            self.params = 9
            self.gridGen = HomographyGridGen(self._h, self._w)
        elif self.geo_model=='tps':
            self.gridGen = TpsGridGen(self._h, self._w, grid_size=tps_grid_size, 
                                      reg_factor=tps_reg_factor)
            self.params = (tps_grid_size**2)*2

        # localization net 
        self.conv1 = nn.Conv2d(in_channels, 32, self._ksize, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, self._ksize, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, self._ksize, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, self._ksize, padding=1, bias=False)

        self.fc1 = nn.Linear(32*self.pooled_size, 1024)
        self.fc2 = nn.Linear(1024, self.params)



    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        b, c, w, h = x.size()

        assert w == self._w and h == self._h, \
            "Given inpus size is diffrent from set params number!"

        batch_images = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        #print("Pre view size:{}".format(x.size()))
        x = x.view(-1, 32*self.pooled_size)
        
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x) # params [Nx6]

        #print(x.size())
        if self.geo_model == 'affine':
            x = x.view(b, 2, 3) # change it to the 52x3 matrix 
        elif self.geo_model == 'hom':
            x = x.view(b, 9)
        elif self.geo_model == 'tps':
            x = x.view(b, -1)

        affine_grid_points = self.gridGen(x)

        assert(affine_grid_points.size(0) == batch_images.size(0)), \
            "The batch sizes of the input images must be same as the generated grid."

        rois = F.grid_sample(batch_images, affine_grid_points)

        return rois, affine_grid_points


if __name__ == '__main__':
    # following code is for test
    import torch

    img = torch.zeros(2, 128, 64, 64).cuda()
    net = SpatialTransformer(128, 64, 3, 'tps', 9).cuda()
    rois, affine_grid_points = net(img)
    print("rois found to be of size:{}".format(rois.size()))