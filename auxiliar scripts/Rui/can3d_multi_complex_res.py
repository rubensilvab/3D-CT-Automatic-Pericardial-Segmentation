import torch
import torch.nn as nn

class CAN3D(nn.Module):
    
    def __init__(self, input_channels=1, output_channels=5):
        
        super().__init__()
        
        self.ConvB1 = self.create_block(input_channels, 3, 3, 1, 1, 'glorot')
        self.ConvB2 = self.create_block(3, 16, 3, (1, 2, 2), 1, 'glorot')
        self.ConvB3 = self.create_block(16, 64, 3, (1, 2, 2), 1, 'glorot')
        self.ConvB4 = self.create_block(64, 64, 3, 1, (1, 2, 2), 'identity')
        self.ConvB5 = self.create_block(64, 64, 3, 1, (1, 5, 5), 'identity')
        self.ConvB6 = self.create_block(64, 64, 3, 1, (1, 9, 9), 'identity')
        self.ConvB7 = self.create_block(64, 64, 3, 1, 1, 'identity')
        self.ConvB8 = self.create_block(64, 256, 3, (1, 2, 2), 1, 'glorot')
        self.ConvB9 = self.create_block(256, 256, 3, 1, (1, 2, 2), 'identity')
        self.ConvB10 = self.create_block(256, 256, 3, 1, (1, 5, 5), 'identity')
        self.ConvB11 = self.create_block(256, 256, 3, 1, (1, 9, 9), 'identity')
        self.ConvB12 = self.create_block(256, 256, 3, 1, 1, 'identity')
        
        self.Deconv1 = self.create_block(256, 64, 3, 1, 1, 'glorot') 
        self.Deconv2 = self.create_block(64, 16, 3, 1, 1, 'glorot')
        self.Deconv3 = self.create_block(16, 4, 3, 1, 1, 'glorot')
        
        self.Upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest') 
        
        self.ConvLast = self.create_conv(6, output_channels, 1, 1, 1, 'binary') 
    
        self.Activation = nn.Softmax()
        
    def create_block(self, in_channels, out_channels, kernel_size, stride, df, init):
        
        block = nn.Sequential(self.create_conv(in_channels, out_channels, kernel_size, stride, df, init),
                              nn.InstanceNorm3d(out_channels, affine=True),
                              nn.LeakyReLU(negative_slope=0.1))
        return block
    
    def create_conv(self, in_channels, out_channels, kernel_size, stride, df, init):
        
        if stride==1:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding='same', padding_mode='replicate', stride=stride, dilation=df)
        else:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=df)
        
        if init=='glorot':
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)            
        elif init=='identity':
            nn.init.dirac_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        elif init=='binary':
            w = torch.tensor([[-1, 0, 0, 0, 1, 1],
                              [0, -1, 0, 0, 1, 1],
                              [0, 0, -1, 0, 1, 1],
                              [0, 0, 0, -1, 1, 1],
                              [0, 0, 0, 0, 0, -1]]).float()
            w = w.reshape(out_channels, 6, 1, 1, 1)
            conv.weight = nn.parameter.Parameter(w)
            nn.init.constant_(conv.bias, 0)
        return conv
    
    def forward(self, x):
        
        x_bin = (x >= 0.13).float()
        
        x = self.ConvB1(x)
        
        x = nn.functional.pad(x, (0,1,0,1,1,1), mode='replicate')
        x = self.ConvB2(x)
        x = nn.functional.pad(x, (0,1,0,1,1,1), mode='replicate')
        x = self.ConvB3(x)
        
        x_skip = x + self.ConvB4(x)
        x_skip = x_skip + self.ConvB5(x_skip)
        x_skip = x_skip + self.ConvB6(x_skip)
        x_skip = self.ConvB7(x_skip)
        
        x = nn.functional.pad(x, (0,1,0,1,1,1), mode='replicate')
        x = self.ConvB8(x)
        
        x = x + self.ConvB9(x)
        x = x + self.ConvB10(x)
        x = x + self.ConvB11(x)
        x = self.ConvB12(x)
        
        x = self.Upsample(x)
        x = self.Deconv1(x)
        
        x = x + x_skip
        
        x = self.Upsample(x)
        x = self.Deconv2(x)
        x = self.Upsample(x)
        x = self.Deconv3(x)
        
        x = torch.cat((x, x_bin, x_bin - 1), dim=1)
        
        x = self.ConvLast(x)
        x = self.Activation(x)
        
        return x