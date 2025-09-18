import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock1D(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock1D, self).__init__()
        f1, f2 = filters

        padding = kernel_size // 2  # Equivalent to 'same' padding for stride=1
        self.conv1 = nn.Conv1d(in_channels, f1, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(f1)

        self.conv2 = nn.Conv1d(f1, f2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(f2)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=2):
        super(ConvBlock1D, self).__init__()
        f1, f2 = filters

        padding = (kernel_size - 1) // 2  # Compute padding manually

        self.conv1 = nn.Conv1d(
            in_channels, f1, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(f1)

        self.conv2 = nn.Conv1d(f1, f2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(f2)

        # Adjust shortcut connection to match dimensions
        self.shortcut = nn.Conv1d(in_channels, f2, kernel_size=1, stride=stride)
        self.bn_shortcut = nn.BatchNorm1d(f2)

    def forward(self, x):
        identity = self.bn_shortcut(self.shortcut(x))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_class=1, input_kernel_size=3, **kwargs):
        super(ResNet1D, self).__init__()

        print(f"input_kernel_size: {input_kernel_size}")
        self.conv1 = nn.Conv1d(
            in_channels,
            64,
            kernel_size=input_kernel_size,
            padding=1,
        )
        self.dropout1 = nn.Dropout(p=0.1)

        self.id_block1 = IdentityBlock1D(64, filters=[64, 64], kernel_size=3)
        self.id_block2 = IdentityBlock1D(64, filters=[64, 64], kernel_size=3)

        self.conv_block = ConvBlock1D(64, filters=[128, 128], kernel_size=3)
        self.id_block3 = IdentityBlock1D(128, filters=[128, 128], kernel_size=3)

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(128, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, num_class)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)

        out = self.id_block1(out)
        out = self.id_block2(out)

        out = self.conv_block(out)
        out = self.id_block3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = self.dropout2(out)

        out = self.fc2(out)

        return out


class WDResNet1D(nn.Module):
    def __init__(self, in_channels, num_class=1):
        super(WDResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, stride=16, kernel_size=64, padding=1)
        self.dropout1 = nn.Dropout(p=0.1)

        self.id_block1 = IdentityBlock1D(64, filters=[64, 64], kernel_size=3)
        self.id_block2 = IdentityBlock1D(64, filters=[64, 64], kernel_size=3)

        self.conv_block = ConvBlock1D(64, filters=[128, 128], kernel_size=3)
        self.id_block3 = IdentityBlock1D(128, filters=[128, 128], kernel_size=3)

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(128, 100)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(100, num_class)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout1(out)

        out = self.id_block1(out)
        out = self.id_block2(out)

        out = self.conv_block(out)
        out = self.id_block3(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = self.dropout2(out)

        out = self.fc2(out)

        return out
