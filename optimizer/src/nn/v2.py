from torch import nn, flatten, unflatten
from torch.nn import functional as F

class _Encoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int],
                 elu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding='same', padding_mode='replicate')
        self.elu = elu
        self.batch_norm = nn.BatchNorm2d(num_features=out_features)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.batch_norm(self.conv(x))
        if self.elu:
            x = F.elu(x)
        return x

class _BottleneckEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding='same', padding_mode='replicate')
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        return F.sigmoid(self.conv(x))

class _Decoder(nn.Module):
    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode='bilinear')

class AutoEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial size is 120x160x3
        self.layers = nn.Sequential(
            _Encoder(3, 16, kernel_size=5, stride=1),
            _Encoder(16, 16, kernel_size=5, stride=1),
            _Encoder(16, 16, kernel_size=5, stride=1),
            _BottleneckEncoder(16, 1, kernel_size=5, stride=1),
            _Encoder(1, 16, kernel_size=5, stride=1),
            _Encoder(16, 16, kernel_size=5, stride=1),
            _Encoder(16, 16, kernel_size=5, stride=1),
            _Encoder(16, 3, kernel_size=5, stride=1),
        )

    def forward(self, x):
        return F.sigmoid(self.layers(x))