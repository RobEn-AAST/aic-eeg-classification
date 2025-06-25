import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


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
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.ELU()
                )
            )
            channels += growth_rate
        self.lff = nn.Conv2d(channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        fused = self.lff(torch.cat(features, dim=1))
        return fused + x


class LFANN_DANN(nn.Module):
    def __init__(self,
                 C,
                 F,
                 T,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_t=64,
                 num_classes=2,
                 reduction=16,
                 domain_classes=30):
        super(LFANN_DANN, self).__init__()
        # Shared feature extractor
        self.conv_t = nn.Sequential(
            nn.Conv2d(C, F1, kernel_size=(1, kernel_t), padding=(0, kernel_t//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(F,1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU()
        )
        self.rdb1 = ResidualDenseBlock(F2, growth_rate=F2//2)
        self.rdb2 = ResidualDenseBlock(F2, growth_rate=F2//2)
        self.cbam = CBAM(F2, reduction)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # Label classifier
        self.classifier = nn.Linear(F2, num_classes)
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(F2, F2//2),
            nn.ELU(),
            nn.Linear(F2//2, domain_classes)
        )

    def forward(self, x, lambd=1.0):
        # x: (batch, C, F, T)
        feat = self.conv_t(x)
        feat = self.depthwise(feat)
        feat = self.pointwise(feat)
        feat = self.rdb1(feat)
        feat = self.rdb2(feat)
        feat = self.cbam(feat)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        # Label prediction
        class_logits = self.classifier(feat)
        # Domain prediction with gradient reversal
        reverse_feat = grad_reverse(feat, lambd)
        domain_logits = self.domain_classifier(reverse_feat)
        return class_logits, domain_logits


if __name__ == "__main__":
    batch, C, F, T = 16, 22, 40, 875
    model = LFANN_DANN(C=C, F=F, T=T)
    inp = torch.randn(batch, C, F, T)
    class_out, domain_out = model(inp, lambd=0.1)
    print("Class output:", class_out.shape)
    print("Domain output:", domain_out.shape)
