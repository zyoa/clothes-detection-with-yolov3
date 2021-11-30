import random
import torch
import torch.nn as nn
import numpy as np

SEED = 18

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_conv(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            make_conv(channels, channels // 2, kernel_size=1, padding=0),
            make_conv(channels// 2, channels , kernel_size=3)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.body = nn.Sequential(
            make_conv(3, 32, kernel_size=3),
            make_conv(32, 64, kernel_size=3, stride=2),
            ResidualBlock(channels=64),
            make_conv(64, 128, kernel_size=3, stride=2),
            ResidualBlock(channels=128),
            ResidualBlock(channels=128),
            make_conv(128, 256, kernel_size=3, stride=2),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            ResidualBlock(channels=256),
            make_conv(256, 512, kernel_size=3, stride=2),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            ResidualBlock(channels=512),
            make_conv(512, 1024, kernel_size=3, stride=2),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
            ResidualBlock(channels=1024),
        )
    
    def forward(self, x):
        return self.body(x)


class FCLayer(nn.Module):
    def __init__(self, num_classes):
        super(FCLayer, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class DarknetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DarknetClassifier, self).__init__()
        self.darknet53 = Darknet53()
        self.fclayer = FCLayer(num_classes)
    
    def forward(self, x):
        x = self.darknet53(x)
        x = self.fclayer(x)
        return x


def train():
    model = DarknetClassifier(1000)
    tensor = torch.rand([1, 3, 1256, 1256])
    print(model(tensor).shape)


if __name__ == '__main__':
    train()
