import torch
import torch.nn as nn


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.block(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(ConvBNReLU1D(channels, growth_rate))
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CDCNBranch(nn.Module):
    def __init__(
        self,
        input_channels,
        growth_rate=16,
        num_blocks=3,
        layers_per_block=4,
        compression=0.5,
    ):
        super().__init__()
        self.initial_conv = nn.Conv1d(
            input_channels, growth_rate * 2, kernel_size=9, padding=4
        )
        channels = growth_rate * 2
        self.blocks = nn.ModuleList()

        for i in range(num_blocks):
            dense = DenseBlock(channels, growth_rate, layers_per_block)
            self.blocks.append(dense)
            channels += growth_rate * layers_per_block
            if i < num_blocks - 1:
                transition = TransitionBlock(channels, compression)
                self.blocks.append(transition)
                channels = int(channels * compression)

        # New part: BN + ReLU before global pooling
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(channels), nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.out_channels = channels

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.bn_relu(x)  # Apply BN + ReLU before pooling
        x = self.global_pool(x)
        return x.squeeze(-1)


class CDCN(nn.Module):
    def __init__(self, in_channels, n_class, branches=1, **kwargs):
        super().__init__()
        self.branch1 = CDCNBranch(in_channels)
        self.branches = branches
        if branches == 2:
            self.branch2 = CDCNBranch(in_channels)
            combined_channels = self.branch1.out_channels + self.branch2.out_channels
            self.classifier = nn.Linear(combined_channels, n_class)
        else:
            self.classifier = nn.Linear(self.branch1.out_channels, n_class)

    def forward(self, x1):  ## customized input to support a single vibration signal
        f1 = self.branch1(x1)
        if self.branches == 2:
            f2 = self.branch2(x1)
            fused = torch.cat([f1, f2], dim=1)
        else:
            fused = f1
        return self.classifier(fused)
