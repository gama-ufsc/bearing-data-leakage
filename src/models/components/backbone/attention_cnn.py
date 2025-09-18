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


class Conv1DPoolBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DPoolBn, self).__init__()
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
        out = self.pool(out)
        out = self.bn(out)
        return out


class Attention1DCNN(nn.Module):
    def __init__(self, in_channels, n_class, factor=16, first_kernel=64):
        super().__init__()
        self.name = "Attention1DCNN"
        self.conv_layers = nn.Sequential(
            ConvBnPool1D(
                in_channels,
                out_channels=16,
                kernel_size=first_kernel,
                stride=16,
                padding=24,
            ),
            ConvBnPool1D(16, 32),
            ConvBnPool1D(32, 64),
            ConvBnPool1D(64, 64),
            ConvBnPool1D(64, 64, padding=0),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fcA = nn.Linear(64, 64 // factor)
        self.fcA1 = nn.Linear(64 // factor, 64)
        self.sigmoid = nn.Sigmoid()

        self.bn = nn.BatchNorm1d(100)
        self.fc = nn.LazyLinear(100)
        self.out = nn.Linear(100, n_class)

    def forward(self, x):
        f1 = self.conv_layers(x)

        # Attention block
        z = self.global_avg_pool(f1)  # (B, C, 1)
        z = z.view(z.size(0), -1)  # (B, C)
        z = F.relu(self.fcA(z))
        z = self.fcA1(z)
        z = F.sigmoid(z)  # (B, C)
        z = z.view(z.size(0), z.size(1), 1)

        f1 = f1 * z  # (B, C, L_pooled), channel-wise reweighting

        f2 = torch.flatten(f1, 1)
        f3 = F.relu(self.bn(self.fc(f2)))
        out = self.out(f3)
        return out


class Attention1DCNN_Custom(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.name = "Attention1DCNN_Custom"

        print(self.name)

        self.conv_layers = nn.Sequential(
            Conv1DPoolBn(
                in_channels, out_channels=16, kernel_size=64, stride=16, padding=24
            ),
            Conv1DPoolBn(16, 32),
            Conv1DPoolBn(32, 64),
            Conv1DPoolBn(64, 64),
            Conv1DPoolBn(64, 64, padding=0),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fcA = nn.Linear(64, 64 // 16)
        self.fcA1 = nn.Linear(64 // 16, 64)
        self.sigmoid = nn.Sigmoid()

        self.bn = nn.BatchNorm1d(100)
        self.fc = nn.LazyLinear(100)
        self.out = nn.Linear(100, n_class)

    def forward(self, x):
        f1 = self.conv_layers(x)

        # Attention block
        z = self.global_avg_pool(f1)  # (B, C, 1)
        z = z.view(z.size(0), -1)  # (B, C)
        z = F.relu(self.fcA(z))
        z = self.fcA1(z)
        z = F.sigmoid(z)  # (B, C)
        z = z.view(z.size(0), z.size(1), 1)

        f1 = f1 * z  # (B, C, L_pooled), channel-wise reweighting

        f2 = torch.flatten(f1, 1)
        f3 = F.relu(self.bn(self.fc(f2)))
        out = self.out(f3)
        return out
