import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs, groups):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs), nn.SiLU(inplace=True),
        )

class UNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg["channels"]
        groups = cfg["norm_groups"]
        self.feature_level = cfg["feature_level"]
        self.encoders = nn.ModuleList([
            ConvBlock(a, b, groups)
            for a, b in zip([cfg["input_channels"]] + channels[:-1], channels)
        ])
        self.decoders = nn.ModuleList([
            ConvBlock(channels[i] + channels[i - 1], channels[i - 1], groups)
            for i in range(len(channels) - 1, 0, -1)
        ])
        self.head = nn.Conv2d(channels[0], cfg["num_classes"], 1)

    def forward(self, image):
        skips = []
        x = image
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        feature = skips[self.feature_level]
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return {"logits": self.head(x), "feature": feature}

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None

class SourceDiscriminator(nn.Module):
    def __init__(self, channels, hidden, grl_scale):
        super().__init__()
        self.grl_scale = grl_scale
        self.classifier = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.SiLU(), nn.Conv1d(hidden, 2, 1)
        )

    def forward(self, feature, valid=None):
        if valid is None:
            columns = feature.mean(dim=2)
        else:
            columns = (feature * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
        return self.classifier(GradientReverse.apply(columns, self.grl_scale))
