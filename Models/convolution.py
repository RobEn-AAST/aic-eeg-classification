import torch
import torch.nn as nn

class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, dim_mult=1, padding=0, bias=False):
        super(DepthWiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * dim_mult, padding=padding, kernel_size=kernel_size, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.depthwise(x)


class SeperableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(SeperableConv2D, self).__init__()
        self.depthwise = DepthWiseConv2D(in_channels, kernel_size, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out