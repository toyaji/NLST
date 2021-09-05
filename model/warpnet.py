from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision.models as models

from model.transformation import GeometricTnf


############################# following codes are mostly come from #############################
# "Convolutional neural network architecture for geometric matching, I. Rocco et al, 2018"
# https://github.com/ignacio-rocco/cnngeometric_pytorch

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)

class FeatureExtraction(torch.nn.Module):
    def __init__(self, train_fe=False, normalization=True, last_layer=''):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.model = models.vgg16(pretrained=True)
        # keep feature extraction network up to indicated layer
        vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                     'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                     'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                     'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                     'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
        if last_layer=='':
            last_layer = 'pool4'
        last_layer_idx = vgg_feature_layers.index(last_layer)
        layers = list(self.model.features.children())[:last_layer_idx+1]
        self.model = nn.Sequential(*layers)
        
        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        if self.normalization:
            x = featureL2Norm(x)
        return x
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True, matching_type='correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type=matching_type
        self.shape=shape
        self.ReLU = nn.ReLU()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        if self.matching_type=='correlation':
            if self.shape=='3D':
                # reshape features for matrix multiplication
                feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
                feature_B = feature_B.view(b,c,h*w).transpose(1,2)
                # perform matrix mult.
                feature_mul = torch.bmm(feature_B,feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            elif self.shape=='4D':
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
                feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
                # perform matrix mult.
                feature_mul = torch.bmm(feature_A,feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
            
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
        
            return correlation_tensor

        if self.matching_type=='subtraction':
            return feature_A.sub(feature_B)
        
        if self.matching_type=='concatenation':
            return torch.cat((feature_A,feature_B),1)

class FeatureRegression(nn.Module):
    def __init__(self, input_size, output_dim=6, batch_normalization=True, channels=[225,128,64,32]):
        super(FeatureRegression, self).__init__()
        num_layers = len(channels)
        # to make adaptive to input size change
        self.connector = nn.Sequential(
            nn.Conv2d(input_size, 225, kernel_size=3, padding=1),
            nn.BatchNorm2d(225),
            nn.ReLU())

        nn_modules = list()
        for i in range(num_layers-1): # last layer is linear 
            ch_in = channels[i]
            ch_out = channels[i+1]            
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU())
        self.conv = nn.Sequential(*nn_modules)

        # lienar map to theata output
        self.linear1 = nn.Linear(ch_out * input_size, 1024)
        self.linear2 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.connector(x)
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
    
class WarpNet(nn.Module):
    def __init__(self, 
                 output_theta=6, 
                 size=1024, 
                 feature_extraction_last_layer='',
                 return_correlation=False,
                 fr_channels=[225, 128, 64, 32],
                 feature_self_matching=False,
                 normalize_features=True,
                 normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,
                 matching_type='correlation'):
        
        super(WarpNet, self).__init__()
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.corr_out_size = int((size/16)**2)

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches,matching_type=matching_type)        
        

        self.FeatureRegression = FeatureRegression(self.corr_out_size,
                                                   output_theta,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU()
        self.transfomer = GeometricTnf('affine', size=size)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, x, y): 
        # feature extraction
        x_features = self.FeatureExtraction(x)
        y_featrues = self.FeatureExtraction(y)
        # feature correlation
        correlation = self.FeatureCorrelation(x_features, y_featrues)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        warped_x = self.transfomer(x, theta)
        return warped_x


if __name__ == '__main__':
    import torch
    img = torch.zeros(3, 3, 1024, 1024).cuda()
    net = WarpNet(size=1024).cuda()
    out = net(img, img)
    print(out.size())

