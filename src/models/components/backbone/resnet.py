"""Modified ResNet model for bearing failure classification."""

from torch import nn


class ModifiedResNet(nn.Module):
    def __init__(
        self, in_channels, num_class, resnet_model, pretrained=False, freeze=False
    ):
        super().__init__()
        self.name = "ModifiedResNet"
        self.resnet = resnet_model

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if not pretrained:
            self.resnet.apply(self._weight_reset)

        # Added first conv layer to handle 2 channel input
        # self.in_conv2d = nn.Conv2d(
        #     in_channels=in_channels, out_channels=3,
        #     kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)
        # )

        # Modified first conv layer for only 1 channel input
        # self.resnet.conv1 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=64,
        #     kernel_size=(7, 7),
        #     stride=(2, 2),
        #     padding=(3, 3),
        #     bias=False,
        # )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_class)

        # Added intermediate layer before fc to analyze features
        # self.resnet.fc = nn.Identity()
        # self.intermediate_fc = nn.Linear(512, 2)
        # self.fc = nn.Linear(2, num_class)

    # def forward(self, X):
    #     # Get features from intermediate layer
    #     backbone_out = self.resnet(X)
    #     features = self.intermediate_fc(backbone_out)
    #     out = self.fc(features)
    #     return out

    def forward(self, X):
        # Layer before input layer
        # cin = self.in_conv2d(X)
        # out = self.resnet(cin)

        # Default
        out = self.resnet(X)
        return out

    def _weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()


class ModifiedResNetTwoOutputs(nn.Module):
    def __init__(self, in_channels, resnet_model, pretrained=False, freeze=False):
        super().__init__()
        self.name = "ModifiedResNetTwoOutputs"
        self.resnet = resnet_model

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if not pretrained:
            self.resnet.apply(self._weight_reset)

        # Modified first conv layer for only 1 channel input
        # self.resnet.conv1 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=64,
        #     kernel_size=(7, 7),
        #     stride=(2, 2),
        #     padding=(3, 3),
        #     bias=False,
        # )

        # Remove fc layer
        self.resnet.fc = nn.Identity()
        self.fc_multilabel = nn.Linear(512, 3)
        self.fc_multiclass = nn.Linear(512, 4)

    def forward(self, X):
        out_shared = self.resnet(X)
        out_multilabel = self.fc_multilabel(out_shared)
        out_multiclass = self.fc_multiclass(out_shared)
        return out_multilabel, out_multiclass

    def _weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()


# for name, p in mod_resnet.named_parameters():
# print(name, p.requires_grad)
