"""Adapters that only use source data. (Baseline without domain adaptation)"""

import torch
from torch import nn


class Baseline(nn.Module):
    def __init__(self, backbone: nn.Module, class_loss: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.class_loss = class_loss

    def forward(
        self, batch: torch.Tensor, stage: str = "inference", **kwargs
    ) -> torch.Tensor:
        X = batch["X"]
        if stage == "inference":
            return self.backbone(X)
        else:
            labels = batch["label"]

            class_preds = self.backbone(X)
            loss = self.class_loss(class_preds, labels)

            return class_preds, loss
