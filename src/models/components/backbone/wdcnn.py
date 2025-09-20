"""
WDCNN model.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvBnPool1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnPool1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out


class WDCNN(nn.Module):
    def __init__(
        self, in_channels, n_class, pretrained, apply_dropout=False, dropout_rate=0.5
    ):
        super().__init__()
        self.name = "WDCNN"
        self.pretrained = pretrained
        self.conv_layers = nn.Sequential(
            ConvBnPool1D(
                in_channels, out_channels=16, kernel_size=64, stride=16, padding=24
            ),
            ConvBnPool1D(16, 32),
            ConvBnPool1D(32, 64),
            ConvBnPool1D(64, 64),
            ConvBnPool1D(64, 64, padding=0),
        )
        self.bn = nn.BatchNorm1d(100)
        self.fc = nn.LazyLinear(100)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.LazyLinear(100)
        self.out = nn.Linear(100, n_class)
        self.apply_dropout = apply_dropout

    def forward(self, x):
        f1 = self.conv_layers(x)
        f2 = torch.flatten(f1, 1)
        f3 = F.relu(self.bn(self.fc(f2)))
        if self.apply_dropout:
            f3 = self.dropout(f3)
        out = self.out(f3)

        return out
