from torch import nn, flatten, unflatten
import torch.nn.functional as F

class _Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = pool
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv(x)))
        if self.pool:
            x = F.avg_pool2d(x, kernel_size=2)
        return x

class _Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = relu
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.batch_norm(self.conv(F.interpolate(x, scale_factor=2, mode='bilinear')))
        if self.relu:
            x = F.relu(x)
        return x

class _Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = relu
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        y = F.relu(self.conv(x))
        x = self.batch_norm(self.conv_2(x + y))
        if self.relu:
            x = F.relu(x)
        return x

class _BottleneckEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = flatten(x, start_dim=1)
        x = F.relu(self.linear(F.sigmoid(x)))
        return x

class _BottleneckDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sizes: list[int]):
        super().__init__()
        self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.sizes = sizes
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = unflatten(x, dim=1, sizes=self.sizes)
        return x

class _BottleneckResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(in_features=channels, out_features=channels)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return F.relu(self.linear(x) + x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _Encoder(3, 16, kernel_size=5), # 58x78x16
            _Encoder(16, 32, kernel_size=5), # 27x37x64
            _Encoder(32, 64, kernel_size=7, pool=False), # 21x31x128
            _Encoder(64, 128, kernel_size=7, pool=False), # 15x25x256
            _Residual(128, 128),
            _Residual(128, 128),
            _Residual(128, 128),
            _Residual(128, 1),
            _BottleneckEncoder(in_channels=375, out_channels=256),
            _BottleneckResidual(channels=256),
            _BottleneckResidual(channels=256),
            _BottleneckResidual(channels=256),
            _BottleneckResidual(channels=256),
            _BottleneckDecoder(in_channels=256, out_channels=500, sizes=[1, 20, 25]),
            _Decoder(1, 16, kernel_size=3), # 38x48x16
            _Residual(16, 16, kernel_size=3), # 36x46x16
            _Decoder(in_channels=16, out_channels=64, kernel_size=3), # 70x90x64
            _Residual(64, 64, kernel_size=3), # 68x88x64
            _Decoder(in_channels=64, out_channels=64, kernel_size=5), # 132x172x32
            _Residual(in_channels=64, out_channels=64, kernel_size=5), # 128x168x64
            _Residual(in_channels=64, out_channels=64, kernel_size=5), # 124x164x64
            _Residual(in_channels=64, out_channels=64, kernel_size=5), # 120x160x64
            _Residual(in_channels=64, out_channels=3, kernel_size=1, relu=False), # 120x160x3
        )

    def forward(self, x):
        return F.sigmoid(self.layers(x))
