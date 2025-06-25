import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False),
            nn.ELU(),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with 4 conv layers, dense connections, local feature fusion.
    """
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.ELU()
                )
            )
            channels += growth_rate
        # Local feature fusion
        self.lff = nn.Conv2d(channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        inputs = x
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.lff(torch.cat(features, dim=1))
        return fused + inputs  # residual connection


class LFANN(nn.Module):
    """
    Lightweight Feature Fusion Network (with RDB) for Motor Imagery EEG Classification.

    Args:
        C (int): Number of EEG channels
        F (int): Number of frequency bins
        T (int): Number of time samples
        F1 (int): Filters in temporal convolution
        D (int): Depth multiplier
        F2 (int): Filters after pointwise conv
        kernel_t (int): Temporal kernel length
        num_classes (int): Output classes
        reduction (int): CBAM reduction ratio
    """
    def __init__(self,
                 C,
                 F,
                 T,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_t=64,
                 num_classes=4,
                 reduction=16):
        super(LFANN, self).__init__()
        # Temporal conv (time axis)
        self.conv_t = nn.Sequential(
            nn.Conv2d(C, F1, kernel_size=(1, kernel_t), padding=(0, kernel_t//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        # Depthwise conv (frequency axis)
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(F,1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU()
        )
        # Pointwise conv
        self.pointwise = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU()
        )
        # Residual Dense Blocks
        self.rdb1 = ResidualDenseBlock(F2, growth_rate=F2//2)
        self.rdb2 = ResidualDenseBlock(F2, growth_rate=F2//2)
        # Attention
        self.cbam = CBAM(F2, reduction)
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x):
        # x: (batch, C, F, T)
        x = self.conv_t(x)       # -> (batch, F1, F, T)
        x = self.depthwise(x)    # -> (batch, F1*D, 1, T)
        x = self.pointwise(x)    # -> (batch, F2, 1, T)
        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.cbam(x)
        x = self.global_pool(x).view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    batch, C, F, T = 16, 22, 40, 875
    model = LFANN(C=C, F=F, T=T)
    inp = torch.randn(batch, C, F, T)
    out = model(inp)
    print("Output shape:", out.shape)
