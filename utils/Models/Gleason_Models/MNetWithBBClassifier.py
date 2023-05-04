import torch
import torch.nn as nn


class BB_classifier(nn.Module):
    def __init__(self, n_input_channels=256, n_features=64, n_output_channels=6, anchor_stride=2, dim=3):
        super(BB_classifier, self).__init__()
        self.n_classes = 6
        self.dim = dim

        self.model = nn.Sequential(
            nn.Conv3d(n_input_channels, n_features, kernel_size=3, stride=anchor_stride, padding=1),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=0.25),
            nn.Conv3d(n_features, n_features, kernel_size=3, stride=anchor_stride, padding=1),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=0.25), 
            nn.Conv3d(n_features, n_features, kernel_size=3, stride=anchor_stride, padding=1),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=0.25),
            nn.Conv3d(n_features, n_features, kernel_size=3, stride=anchor_stride, padding=1),
            nn.BatchNorm3d(n_features),
            nn.ReLU(inplace=True),
            #nn.Dropout3d(p=0.25),
            nn.Conv3d(n_features, n_output_channels, kernel_size=3, stride=anchor_stride, padding=1)
        )

    def forward(self, x):
        x = self.model(x)

        # Rearrange dimensions based on self.dim
        if self.dim == 2:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size()[0], -1, self.n_classes)
        else:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size()[0], -1, self.n_classes)

        # Apply softmax activation
        x = nn.functional.softmax(x[0], dim=1)

        return x
    

from torch.nn.modules.conv import Conv3d
import torch
import torch.nn as nn



def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5)
        )  

class MNetWithBBClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
                
        self.dconv_down1 = double_conv(4, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool3d((2,2,2), stride=(2,2,2))
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv3d(64,num_classes, 1)

        self.e1_e3 = nn.Conv3d(64, 256, kernel_size=3, stride=(2,2,2), padding=1)
        self.conv_mid2 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)

        self.e2_e4 = nn.Conv3d(128, 512, kernel_size=3, stride=(2,2,2), padding=1)
        self.conv_mid1 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.bb_classifier = BB_classifier()
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) 
        
        x = self.dconv_down4(x)

        mid = self.e2_e4(conv2)
        mid = self.maxpool(mid)
     
        x = torch.cat([mid,x], dim=1)
        x = self.conv_mid1(x)


        x = self.upsample(x)

        mid = self.e1_e3(conv1)
        mid = self.maxpool(mid)
        mid = torch.cat([conv3,mid], dim=1)
        mid = self.conv_mid2(mid)

        x = torch.cat([x, mid], dim=1)
        
        x = self.dconv_up3(x)
        bb_output = self.bb_classifier(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv2], dim=1)       

        # x = self.dconv_up2(x)
        # x = self.upsample(x)   
        # x = torch.cat([x, conv1], dim=1)   
        
        # x = self.dconv_up1(x)
        
        # out = self.conv_last(x)
        
        return bb_output
    


