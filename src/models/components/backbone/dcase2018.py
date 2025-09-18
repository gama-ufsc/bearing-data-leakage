import torch.nn as nn


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Conv1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # If input and output channels are different, apply 1x1 conv to residual
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        # Match residual shape if needed
        if self.residual_conv:
            residual = self.residual_conv(residual)
        out += residual  # Residual Connection
        out = self.relu(out)
        return out


class DCASE2018(nn.Module):
    def __init__(self, n_class=1, **kwargs):
        super(DCASE2018, self).__init__()

        self.name = "DCASE2018"
        self.initial_conv = nn.Conv1d(1, 48, kernel_size=80, stride=4, padding=40)
        self.initial_bn = nn.BatchNorm1d(48)
        self.initial_relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block1 = Conv1DBlock(48, 48)
        self.block2 = Conv1DBlock(48, 48)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block3 = Conv1DBlock(48, 96)
        self.block4 = Conv1DBlock(96, 96)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block5 = Conv1DBlock(96, 192)
        self.block6 = Conv1DBlock(192, 192)
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.block7 = Conv1DBlock(192, 384)
        self.block8 = Conv1DBlock(384, 384)
        self.pool5 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(384, n_class)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool3(x)

        x = self.block5(x)
        x = self.block6(x)
        x = self.pool4(x)

        x = self.block7(x)
        x = self.block8(x)
        x = self.pool5(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
