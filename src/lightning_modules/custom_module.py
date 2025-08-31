from typing import Mapping, Optional, Type

import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassStatScores,
)

from src.lightning_modules.base_module import BaseModule
from src.utils.seg_losses import SoftDiceBCELoss


class Segmentation(BaseModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_labels: int,
        loss: torch.nn.Module,
        optim_class: Optional[Type] = None,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        log_lr: bool = True,
        log_grad_norm: bool = False,
        sync_dist: bool = False,  # if ``True``, reduces the metric across devices. Causes overhead. Use only for multi-gpu train
        plot_dict: Optional[Mapping] = None,
    ):
        super().__init__()

        self.model = model
        self.loss = loss
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()
        self.log_lr = log_lr
        self.log_grad_norm = log_grad_norm
        self.sync_dist = sync_dist
        self.plot_preds_at_epoch = plot_dict

        if isinstance(loss, SoftDiceBCELoss):
            # Binary metrics
            self.train_metrics = torchmetrics.MetricCollection(
                {
                    "train_f1": torchmetrics.F1Score(task="binary"),
                    "train_iou": torchmetrics.JaccardIndex(task="binary"),
                }
            )
            self.val_metrics = torchmetrics.MetricCollection(
                {
                    "val_f1": torchmetrics.F1Score(task="binary"),
                    "val_iou": torchmetrics.JaccardIndex(task="binary"),
                }
            )
            self.register_buffer(
                "prob_map", torch.tensor([0.0, 0.95, 0.75, 0.3], dtype=torch.float32)
            )
        else:
            # Multiclass metrics
            self.train_metrics = torchmetrics.MetricCollection(
                {
                    "train_f1": MulticlassF1Score(num_classes=num_labels, average="macro"),
                    "train_scores": MulticlassStatScores(num_classes=num_labels, average="macro"),
                    "train_miou": MulticlassJaccardIndex(num_classes=num_labels, average="macro"),
                }
            )
            self.val_metrics = torchmetrics.MetricCollection(
                {
                    "val_f1": MulticlassF1Score(num_classes=num_labels, average="macro"),
                    "val_scores": MulticlassStatScores(num_classes=num_labels, average="macro"),
                    "val_miou": MulticlassJaccardIndex(num_classes=num_labels, average="macro"),
                }
            )

    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.forward(x)

        if isinstance(self.loss, SoftDiceBCELoss):
            # --- Binary with soft labels ---
            y = y.long()
            soft_targets = self.prob_map[y].unsqueeze(1).float()
            loss = self.loss(y_pred, soft_targets)

            # metrics: use hard predictions
            pred_labels = (torch.sigmoid(y_pred) > 0.5).int()
            target_labels = (soft_targets > 0.5).int()
            self.train_metrics.update(pred_labels.detach(), target_labels)

        else:
            # --- Multiclass case (as before) ---
            loss = self.loss(y_pred, y.unsqueeze(1).long())
            self.train_metrics.update(y_pred.argmax(1).detach().int(), y.int())

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )

        return {"loss": loss}

    # - - - - Validation - - - - - - - - - - - - - - - - - - - - #

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.forward(x)

        if isinstance(self.loss, SoftDiceBCELoss):
            # --- Binary with soft labels ---
            y = y.long()
            soft_targets = self.prob_map[y].unsqueeze(1).float()
            loss = self.loss(y_pred, soft_targets)

            # metrics: use hard predictions for evaluation
            pred_labels = (torch.sigmoid(y_pred) > 0.5).int()
            target_labels = (soft_targets > 0.5).int()
            self.val_metrics.update(pred_labels, target_labels)

        else:
            # --- Multiclass setup (your original flow) ---
            loss = self.loss(y_pred, y.unsqueeze(1).long())
            self.val_metrics.update(y_pred.argmax(1).int(), y.int())

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )

        return {"loss": loss}
