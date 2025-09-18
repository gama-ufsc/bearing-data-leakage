import inspect
import os
from typing import Any, Dict, Optional, Callable

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from src.data.transforms.segmentation import segment_signals


def set_requires_grad(net, requires_grad=False):
    """
    Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    """
    for param in net.parameters():
        param.requires_grad = requires_grad


class BaseModelModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics_params: Dict[str, Any],
        project: str,
        experiment: str,
        seed: int,
        save_outputs: bool = False,
        pretrain_epochs: Optional[int] = None,
        cfg_dict: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
        ] = None,
        ensembled_eval: bool = False,
        classification_method: str = "multilabel",
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.metrics_params = metrics_params
        self.project = project
        self.experiment = experiment
        self.seed = seed
        self.save_outputs = save_outputs
        self.pretrain_epochs = pretrain_epochs
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.cfg_dict = cfg_dict
        self.lr_scheduler_cfg = lr_scheduler
        self.patience = 2
        self.ensembled_eval = ensembled_eval
        self.classification_method = classification_method

        # TODO: Refactor to avoid saving PyTorch NN modules (model)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if (self.pretrain_epochs is not None) and (
            self.pretrain_epochs >= (self.current_epoch + 1)
        ):
            stage = "pretrain"
        else:
            stage = "train"

        batch = self._adjust_batch_label(batch)
        preds, loss = self.model(
            batch,
            stage=stage,
        )

        self.train_outputs.append(
            {"idx": batch["idx"], "labels": batch["label"], "preds": preds}
        )

        # Log LR from the first param group
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(f"{stage}/lr", current_lr, on_epoch=True, prog_bar=False)

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._adjust_batch_label(batch)

        if not self.ensembled_eval:
            preds, loss = self.model(
                batch,
                stage="val",
            )
            self.val_outputs.append(
                {"idx": batch["idx"], "labels": batch["label"], "preds": preds}
            )
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        else:
            # Extract original inputs
            inputs = batch[
                "X"
            ]  # assuming batch["input"] has shape [batch_size, 1, 255000]
            labels = batch["label"]  # [batch_size]
            idxs = batch["idx"]  # [batch_size]

            # === SEGMENT INPUT ===
            segment_size = self.cfg_dict["crop_size"]
            overlap = self.cfg_dict["overlap_pct"]  # True or False
            segments = segment_signals(
                inputs, segment_size=segment_size, overlap_ratio=overlap
            )
            # Now segments shape is [batch_size, 1, n_segments, segment_size]

            batch_size, _, n_segments, segment_size = segments.shape
            segments = segments.permute(0, 2, 1, 3).reshape(
                batch_size * n_segments, 1, segment_size
            )
            # Now shape: [batch_size * n_segments, 1, segment_size]

            # Expand labels to match segments
            labels_segments = (
                labels.unsqueeze(1).expand(batch_size, n_segments, 2).reshape(-1, 2)
            )
            idxs_segments = idxs.unsqueeze(1).expand(batch_size, n_segments).reshape(-1)

            # === PREDICT ON SEGMENTS ===
            preds_segments, loss = self.model(
                {"X": segments, "label": labels_segments, "idx": idxs_segments},
                stage="val",
            )
            # preds_segments shape: [batch_size * n_segments, num_classes] (or whatever output)

            # === ENSEMBLE SEGMENT PREDICTIONS PER ORIGINAL SAMPLE ===
            probs_segments = torch.sigmoid(preds_segments)
            probs_segments = probs_segments.view(
                batch_size, n_segments, -1
            )  # [batch_size, n_segments, num_classes]

            # Ensemble probabilities
            probs_ensembled = probs_segments.mean(
                dim=1
            )  # final output: [batch_size, num_classes]

            # === SAVE RESULTS ===
            self.val_outputs.append(
                {"idx": idxs, "labels": labels, "preds": probs_ensembled}
            )
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

            return loss

    def test_step(self, batch, batch_idx):
        batch = self._adjust_batch_label(batch)
        preds, loss = self.model(
            batch,
            stage="test",
        )
        self.test_outputs.append(
            {"idx": batch["idx"], "labels": batch["label"], "preds": preds}
        )

        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        if (self.pretrain_epochs is not None) and (
            self.pretrain_epochs >= (self.current_epoch + 1)
        ):
            stage = "pretrain"
            if (self.current_epoch + 1) == self.pretrain_epochs:
                # Update weights of the target feature extractor
                self.model.target_feature_extractor.load_state_dict(
                    self.model.source_feature_extractor.state_dict()
                )
                # Freeze the source feature extractor and classifier
                set_requires_grad(
                    self.model.source_feature_extractor, requires_grad=False
                )
                set_requires_grad(self.model.classifier, requires_grad=False)
        else:
            stage = "train"

        idxs, preds, labels = self._shared_epoch_end(self.train_outputs, stage=stage)

        if self.save_outputs:
            if len(preds.shape) != len(labels.shape):
                labels = labels.unsqueeze(1)
            merged_output_tensor = torch.cat((idxs.unsqueeze(1), preds, labels), dim=1)
            if not os.path.exists(f"data/artifacts/logits/{self.experiment}/train"):
                os.makedirs(f"data/artifacts/logits/{self.experiment}/train")

            torch.save(
                merged_output_tensor,
                f"data/artifacts/logits/{self.experiment}/train/epoch_{self.current_epoch}_run_{self.seed}.pt",
            )
        self.train_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        idxs, preds, labels = self._shared_epoch_end(self.val_outputs, stage="val")
        self.val_outputs.clear()

        # Manually step ReduceLROnPlateau after warmup
        if self.lr_schedulers():
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, dict) and "scheduler" in scheduler:
                scheduler = scheduler["scheduler"]

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.current_epoch >= self.patience:  # your warmup_epochs
                    scheduler.step(self.trainer.callback_metrics["val/AUROC"])
                else:
                    print(
                        "Not stepping ReduceLROnPlateau scheduler because we're still in warmup"
                    )

    def on_test_epoch_end(self) -> None:
        idxs, preds, labels = self._shared_epoch_end(self.test_outputs, stage="test")
        if self.save_outputs:
            if len(preds.shape) != len(labels.shape):
                labels = labels.unsqueeze(1)
            merged_output_tensor = torch.cat((idxs.unsqueeze(1), preds, labels), dim=1)
            if not os.path.exists(f"data/artifacts/logits/{self.experiment}/test"):
                os.makedirs(f"data/artifacts/logits/{self.experiment}/test")

            torch.save(
                merged_output_tensor,
                f"data/artifacts/logits/{self.experiment}/test/epoch_{self.current_epoch}_run_{self.seed}.pt",
            )

        self.test_outputs.clear()

    def on_test_start(self) -> None:
        if hasattr(self.model, "update_batch_norm_params"):
            self.model.update_batch_norm_params(
                self.trainer.datamodule.test_dataloader()
            )

    def configure_optimizers(self):
        optimizer_params = self._get_optimizer_params()
        self._clean_optimizer_keywords()
        optimizer = self.optimizer(optimizer_params)

        if self.lr_scheduler_cfg is not None:
            scheduler = self.lr_scheduler_cfg(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "manual",  # dummy name
                    "strict": False,
                },
            }

        return optimizer

    def _get_optimizer_params(self):
        params = []
        default_lr = self.optimizer.keywords.get("lr")

        if hasattr(self.model, "source_feature_extractor") and hasattr(
            self.model, "target_feature_extractor"
        ):
            params.extend(
                self._get_component_params("source_feature_extractor", default_lr)
            )
            params.extend(
                self._get_component_params("target_feature_extractor", default_lr)
            )
        elif hasattr(self.model, "feature_extractor"):
            params.extend(self._get_component_params("feature_extractor", default_lr))

        if hasattr(self.model, "classifier"):
            params.extend(self._get_component_params("classifier", default_lr * 10))

        if hasattr(self.model, "domain_classifier"):
            params.extend(
                self._get_component_params("domain_classifier", default_lr * 10)
            )

        if hasattr(self.model, "backbone"):
            params.extend(self._get_component_params("backbone", default_lr))

        return params

    def _get_component_params(self, component_name, default_lr):
        lr_key = f"{component_name}_lr"
        lr = self.optimizer.keywords.get(lr_key, default_lr)
        component = getattr(self.model, component_name)
        return [{"params": component.parameters(), "lr": lr}]

    def _clean_optimizer_keywords(self):
        valid_params = set(
            inspect.getfullargspec(self.optimizer.__init__).args
            + inspect.getfullargspec(self.optimizer.__init__).kwonlyargs
        )
        keys = {**self.optimizer.keywords}
        # Optimizer keywords is a readonly attribute:
        for k in keys:
            if k not in valid_params:
                print(f"Removing invalid optimizer keyword: {k}")
                self.optimizer.keywords.pop(k)

    def _adjust_batch_label(self, batch):
        if len(batch["label"].shape) > 1:
            # Assume is a multiclass classification problem shape is (batch_size, 1) and we need (batch_size)
            if (
                batch["label"].shape[1] == 1
                and self.classification_method == "multiclass"
            ):
                # Reshape and convert to long (expected by CrossEntropyLoss)
                batch["label"] = batch["label"].reshape(-1).long()
        return batch

    def _shared_epoch_end(
        self,
        outputs,
        stage,
    ):
        preds = torch.cat([out["preds"] for out in outputs])
        labels = torch.cat([out["labels"] for out in outputs])
        idxs = torch.cat([out["idx"] for out in outputs])

        eval_metric = self.metrics_params["eval_metric"](
            preds.cpu(), labels.long().cpu()
        )
        self.log(
            f"{stage}/{self.metrics_params['eval_metric_name']}",
            eval_metric,
            on_epoch=True,
            prog_bar=True,
        )

        if self.metrics_params.get("additional_metrics", None) is not None:
            for metric_dict in self.metrics_params["additional_metrics"]:
                metric_fn = metric_dict["metric"]
                metric_name = metric_dict["metric_name"]
                if isinstance(metric_fn, MulticlassAccuracy) and isinstance(
                    metric_name, list
                ):
                    if metric_fn.average == "macro":
                        metric = torch.tensor(
                            [
                                metric_fn(
                                    torch.sigmoid(preds[:, i].cpu()) > 0.5,
                                    labels[:, i].long().cpu(),
                                )
                                for i in range(labels.shape[1])
                            ]
                        )

                else:
                    metric = metric_fn(preds.cpu(), labels.long().cpu())
                if len(metric.shape) != 0:
                    if metric.shape[0] > 1:
                        try:
                            assert metric.shape[0] == len(metric_name)
                        except AssertionError:
                            raise ValueError from AssertionError(
                                "When returning a metric list, the metric name must be a list of the same length."
                            )

                        for i, m in enumerate(metric):
                            self.log(
                                f"{stage}/{metric_name[i]}",
                                m,
                                on_epoch=True,
                                prog_bar=True,
                            )
                else:
                    self.log(
                        f"{stage}/{metric_name}", metric, on_epoch=True, prog_bar=True
                    )

        return idxs, preds, labels
