import torch.nn.functional as F
from torch import nn, concat

class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='replicate')
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = relu
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        if self.relu:
            return F.relu(x)
        else:
            return x

class Network(nn.Module):
    def __init__(self, n: int = 1):
        super().__init__()

        self.e1 = nn.Sequential(_Block(     3,  8 * n), _Block( 8 * n,  8 * n))
        self.e2 = nn.Sequential(_Block( 8 * n, 16 * n), _Block(16 * n, 16 * n))
        self.e3 = nn.Sequential(_Block(16 * n, 32 * n), _Block(32 * n, 32 * n))
        self.e4 = nn.Sequential(_Block(32 * n, 64 * n), _Block(64 * n, 64 * n))

        self.d1 = nn.Sequential(_Block(64 * n, 32 * n), _Block(32 * n, 32 * n))
        self.d2 = nn.Sequential(_Block(64 * n, 16 * n), _Block(16 * n, 16 * n))
        self.d3 = nn.Sequential(_Block(32 * n,  8 * n), _Block( 8 * n,  8 * n))
        self.d4 = nn.Sequential(_Block(16 * n,  4 * n, relu=False), _Block( 4 * n,      3, relu=False))

    def forward(self, x):
        x = self.e1(x)
        y = x
        x = self.e2(F.max_pool2d(x, kernel_size=2)) # 240x320
        z = x
        x = self.e3(F.max_pool2d(x, kernel_size=2)) # 120x160
        w = x
        x = self.e4(F.max_pool2d(x, kernel_size=2)) # 60x80

        # Upsampling
        x = F.interpolate(self.d1(x), scale_factor=2, mode='nearest-exact') # 120x160
        x = concat((x, w), dim=1)

        x = F.interpolate(self.d2(x), scale_factor=2, mode='nearest-exact') # 240x320
        x = concat((x, z), dim=1)

        x = F.interpolate(self.d3(x), scale_factor=2, mode='nearest-exact') # 480x640
        x = concat((x, y), dim=1)

        return F.sigmoid(self.d4(x))