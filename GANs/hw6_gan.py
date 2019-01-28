# defn for generator and discriminator and disc2

import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Sequential(
             nn.Linear(100, 196*4*4),
             nn.BatchNorm1d(196*4*4)
        )
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196) 
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196) 
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(196)
        )
        # no relu or batch norm, use hyperboblic tangent; during training, real images will be scaled btw -1 to 1.
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

       
    def forward(self, x):
        out = self.fc1(x)        
        out = out.reshape(out.size(0), 196,4,4)

        out = self.layer1(out)        
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


'''
# also for the synthetic feature part

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.MaxPool = nn.MaxPool2d(4, stride=4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        self.ln1 = nn.LayerNorm((196, 32, 32))
        self.ln2 = nn.LayerNorm((196, 16, 16))
        self.ln3 = nn.LayerNorm((196, 16, 16))
        self.ln4 = nn.LayerNorm((196, 8, 8))
        self.ln5 = nn.LayerNorm((196, 8, 8))
        self.ln6 = nn.LayerNorm((196, 8, 8))
        self.ln7 = nn.LayerNorm((196, 8, 8))
        self.ln8 = nn.LayerNorm((196, 4, 4))



#    def forward(self, x, extract_features=0):
        # 1
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.ln1(out)
        # 2
        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.ln2(out)

        # 3
        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.ln3(out)

        # 4
        out = self.conv4(out)
        out = F.leaky_relu(out)
        out = self.ln4(out)

        if extract_features == 4:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h
        # 5
        out = self.conv5(out)
        out = F.leaky_relu(out)
        out = self.ln5(out)

        # 6
        out = self.conv6(out)
        out = F.leaky_relu(out)
        out = self.ln6(out)

        # 7
        out = self.conv7(out)
        out = F.leaky_relu(out)
        out = self.ln7(out)

        # 8
        out = self.conv8(out)
        out = F.leaky_relu(out)
        out = self.ln8(out)

        if extract_features == 8:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h

        out = self.MaxPool(out)

        out = out.view(out.size(0), -1)
        out_fc1 = self.fc1(out)
        out_fc10 = self.fc10(out)

        return out_fc1, out_fc10
'''


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,32,32])
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,16,16]) 
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,16,16])
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,8,8])
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,8,8])
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,8,8]) 
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,8,8])
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([196,4,4]),
#            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.MaxPool = nn.MaxPool2d(4, stride=4)
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196, 10)    

# Original disc
#    def forward(self, x):

# part 2 disc
    def forward(self, x, extract_features=0):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if extract_features == 4:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        if extract_features == 8:
            h = F.max_pool2d(out, 4, 4)
            h = h.view(out.size(0), -1)
            return h

        out = self.MaxPool(out)

        out = out.reshape(out.size(0), -1)

        out1 = self.fc1(out)
        out10 = self.fc10(out)
        return out1, out10


