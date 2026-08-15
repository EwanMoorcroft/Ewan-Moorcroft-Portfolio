"""ResNet18 construction."""

from __future__ import annotations


def build_resnet18(class_count: int, dropout: float, pretrained: bool):
    from torch import nn
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    network = models.resnet18(weights=weights)
    feature_count = network.fc.in_features
    network.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_count, class_count))
    return network
