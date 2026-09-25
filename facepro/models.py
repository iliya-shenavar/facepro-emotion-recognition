from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.drop = nn.Dropout2d(0.05)

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        return F.relu(out + identity, inplace=True)


class EmotionResCNN(nn.Module):
    def __init__(self, num_classes: int = 7, width: int = 48, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            ResidualBlock(width, width * 2, stride=1),
            ResidualBlock(width * 2, width * 2),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.layer3 = nn.Sequential(
            ResidualBlock(width * 2, width * 4, stride=1),
            ResidualBlock(width * 4, width * 4),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.layer4 = nn.Sequential(
            ResidualBlock(width * 4, width * 8, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, width * 4),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.classifier(x)


class SmallEmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _build_resnet18(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(m.fc.in_features, num_classes),
    )
    return m


def _build_efficientnet_b0(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.features[-3:].parameters():
            p.requires_grad = True
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes),
    )
    return m


def _build_mobilenet_v3(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.features[-4:].parameters():
            p.requires_grad = True
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    name = name.lower().strip()
    if name == "small":
        return SmallEmotionCNN(num_classes=num_classes)
    if name in {"rescnn", "res_cnn", "emotion_res_cnn"}:
        return EmotionResCNN(num_classes=num_classes)
    if name == "resnet18":
        return _build_resnet18(num_classes, pretrained, freeze_backbone)
    if name in {"efficientnet_b0", "efficientnet"}:
        return _build_efficientnet_b0(num_classes, pretrained, freeze_backbone)
    if name in {"mobilenet_v3", "mobilenet"}:
        return _build_mobilenet_v3(num_classes, pretrained, freeze_backbone)
    raise ValueError(
        f"Unknown model: {name}. Use 'small', 'rescnn', 'resnet18', 'efficientnet_b0', or 'mobilenet_v3'."
    )
