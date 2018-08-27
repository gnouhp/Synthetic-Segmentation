import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    
    def __init__(self, n_classes, img_size=256):    
        super(FCN, self).__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        self.samepool = nn.AvgPool2d(kernel_size=5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(inplace=True)

        self.sameconv1 = nn.Conv2d(3, 8, 7, padding=3, stride=1)
        self.sameconv2 = nn.Conv2d(8, 16, 5, padding=2, stride=1)
        self.sameconv3 = nn.Conv2d(16, n_classes+1, 5, padding=2, stride=1)

    def forward(self, x):
        x = self.samepool(self.relu(self.sameconv1(x)))
        x = self.samepool(self.relu(self.sameconv2(x)))
        x = self.samepool(self.relu(self.sameconv3(x)))
        return F.log_softmax(x.view(-1, self.n_classes+1, self.img_size, self.img_size), dim=1)
