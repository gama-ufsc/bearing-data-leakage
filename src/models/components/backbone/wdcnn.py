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

        # if output = [batch_size, 1], return [batch_size]
        # if out.shape[1] == 1:
        #    out = out.squeeze(1)
        return out


class Baseline1DCNN(nn.Module):
    def __init__(self, input_length=2048, n_class=3, in_channels=1, pretrained=False):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        # Calculate output length after 4 poolings
        conv_output_len = input_length // (2**4)  # 2048 / 16 = 128
        self.fc = nn.Linear(128 * conv_output_len, n_class)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        return self.fc(x)
