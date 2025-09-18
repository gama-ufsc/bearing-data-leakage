import torch
import torch.nn as nn
import torch.nn.functional as F


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
            groups=in_channels,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out


class WDTCNN(nn.Module):
    def __init__(self, n_class=4, input_size=42000, **kwargs):
        super(WDTCNN, self).__init__()

        def same_padding(kernel_size, stride=1):
            return max(
                (kernel_size - stride) // 2, 0
            )  # Ensures correct padding for different stride values

        self.name = "WDTCNN"
        print(self.name)

        self.conv_layers = nn.Sequential(
            ConvBnPool1D(
                1, 32, kernel_size=64, stride=16, padding=same_padding(64, stride=16)
            ),
            ConvBnPool1D(32, 32, kernel_size=3, stride=1, padding=same_padding(3)),
            ConvBnPool1D(32, 64, kernel_size=3, stride=1, padding=same_padding(3)),
            ConvBnPool1D(64, 64, kernel_size=3, stride=1, padding=same_padding(3)),
            ConvBnPool1D(64, 64, kernel_size=3, stride=1, padding=0),
        )

        # Use dummy input to infer the final flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)  # batch_size=1, channels=1
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flattened_size, 100)  # Adjuste for input size = 42k
        self.out = nn.Linear(100, n_class)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.out(x)

        return x
