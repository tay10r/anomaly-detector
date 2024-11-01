import torch.nn.functional as F
from torch import nn, concat

class _Block(nn.Module):
    def __init__(self, n: int, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * n, out_channels * n, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels * n)
        self.relu = relu
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        if self.relu:
            return F.leaky_relu(x)
        else:
            return x

class Network(nn.Module):
    def __init__(self, n: int = 1):
        super().__init__()
        self.e1 = nn.Sequential(
            _Block(n=1, in_channels=3, out_channels=n*8, kernel_size=7),
            _Block(n, 8, 16, kernel_size=7)
        )
        self.e2 = nn.Sequential(
            _Block(n=1, in_channels=3, out_channels=n*8, kernel_size=5),
            _Block(n, 8, 8, kernel_size=3),
            _Block(n, 8, 16, kernel_size=3),
            _Block(n, 16, 16, kernel_size=3),
            _Block(n, 16, 16, kernel_size=3),
        )
        self.e3 = nn.Sequential(
            _Block(n, 32, 32, kernel_size=7),
            _Block(n, 32, 64, kernel_size=7),
            _Block(n, 64, 64, kernel_size=3)
        )
        self.e4 = nn.Sequential( # 54
            _Block(n, 32, 32, kernel_size=3),
            _Block(n, 32, 32, kernel_size=3),
            _Block(n, 32, 64, kernel_size=3),
            _Block(n, 64, 64, kernel_size=3),
            _Block(n, 64, 64, kernel_size=3),
            _Block(n, 64, 64, kernel_size=3),
            _Block(n, 64, 64, kernel_size=3)
        )
        self.ef = nn.Sequential(
            _Block(n, 128, 128, kernel_size=1),
            _Block(n, 128, 64, kernel_size=1),
            _Block(n, 64, 64, kernel_size=1),
            _Block(n, 64, 16, kernel_size=1),
            _Block(n=1, in_channels=16*n, out_channels=3, kernel_size=1, relu=False)
        )

    def forward(self, x):
        y = F.max_pool2d(self.e1(x), kernel_size=2)
        z = F.max_pool2d(self.e2(x), kernel_size=2)
        x = concat((y, z), dim=1)
        y = self.e3(x)
        z = self.e4(x)
        x = concat((y, z), dim=1)
        x = F.sigmoid(self.ef(x))
        return x
