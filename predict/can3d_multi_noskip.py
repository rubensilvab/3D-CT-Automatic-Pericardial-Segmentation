import torch
import torch.nn as nn

class CAN3D(nn.Module):
    
    def __init__(self, input_channels=1, output_channels=1):
        
        super().__init__()
        
        self.ConvB1 = self.create_block(input_channels, 3, 3, 1, 1, 'glorot')
        self.ConvB2 = self.create_block(3, 16, 3, (1, 2, 2), 1, 'glorot')
        self.ConvB3 = self.create_block(16, 64, 3, (1, 2, 2), 1, 'glorot')
        self.ConvB4 = self.create_block(64, 64, 3, 1, (1, 2, 2), 'identity')
        self.ConvB5 = self.create_block(64, 64, 3, 1, (1, 5, 5), 'identity')
        self.ConvB6 = self.create_block(64, 64, 3, 1, (1, 9, 9), 'identity')
        self.ConvB7 = self.create_block(64, 64, 3, 1, 1, 'identity')
        
        self.Deconv2 = self.create_block(64, 16, 3, 1, 1, 'glorot')
        self.Deconv3 = self.create_block(16, 4, 3, 1, 1, 'glorot')
        
        self.Upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest') 
        
        self.ConvLast = self.create_conv(4, output_channels, 1, 1, 1, 'glorot') 
    
        self.Activation = nn.Sigmoid()
        
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
    
    def Reconvert2HU(self,x,img_min=-1000,img_max=1000):
        img=x*(img_max-img_min)+img_min
          
        return img.float()
    
    def FilterHU(self, img,img_min=-125,img_max=225):
        
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        
        normalize=(img-img_min)/(img_max-img_min)
        
        return normalize.float()
    
    def forward(self, x):
        
       #x_bin = (x >= 0.13).float()
        #x_HU=self.Reconvert2HU(x)  
        #x_peri=self.FilterHU(x_HU)
        #print(x_peri.shape)
        #print(x.shape)
        
        x = self.ConvB1(x)
        
        x = nn.functional.pad(x, (0,1,0,1,1,1), mode='replicate')
        x = self.ConvB2(x)
        x = nn.functional.pad(x, (0,1,0,1,1,1), mode='replicate')
        x = self.ConvB3(x)
        
        x = x + self.ConvB4(x)
        x = x + self.ConvB5(x)
        x = x + self.ConvB6(x)
        x = self.ConvB7(x)
        
        x = self.Upsample(x)
        x = self.Deconv2(x)
        x = self.Upsample(x)
        x = self.Deconv3(x)
        
        #x = torch.cat((x, x_peri), dim=1)
        
        x = self.ConvLast(x)
        x = self.Activation(x)
        
        return x
    
# if torch.cuda.is_available():
#     my_device = torch.device('cuda')
# else:
#     my_device = torch.device('cpu')
# print('Device: {}'.format(my_device))    
    
# model=CAN3D().to(my_device)

# from torchsummary import summary

# summary(model,(1,64,256,256),batch_size = -1)  