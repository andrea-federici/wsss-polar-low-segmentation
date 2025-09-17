import os
from typing import Mapping, Optional, Type

import cv2
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassStatScores,
)

from src.lightning_modules.base_module import BaseModule


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
        self.num_labels = num_labels

        if num_labels == 2:
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

    def load_gt_mask(self, mask_folder: str, filename: str):
        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask == 255).astype(np.uint8)
        return mask

    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.forward(x)
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
        loss = self.loss(y_pred, y.unsqueeze(1).long())
        self.val_metrics.update(y_pred.argmax(1).int(), y.int())

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )

        return {"loss": loss}

    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def on_train_epoch_end(self):
        f1 = self.train_metrics["train_f1"].compute()

        if self.num_labels == 2:
            iou = self.train_metrics["train_iou"].compute()

            train_dict = {
                "train_f1": f1,
                "train_iou": iou,
            }
        else:
            tp, fp, tn, fn, _ = self.train_metrics["train_scores"].compute()
            miou = self.train_metrics["train_miou"].compute()
            train_dict = {
                "train_f1": f1,
                "train_tp": tp.float(),
                "train_fp": fp.float(),
                "train_tn": tn.float(),
                "train_fn": fn.float(),
                "train_iou": tp.float() / (tp + fp + fn).float(),
                "train_mean_iou": miou,
            }

        self.log_dict(train_dict, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    # - - - - Validation - - - - - - - - - - - - - - - - - - - - #

    def on_validation_epoch_end(self):
        f1 = self.val_metrics["val_f1"].compute()

        if self.num_labels == 2:
            iou = self.val_metrics["val_iou"].compute()

            val_dict = {
                "val_f1": f1,
                "val_iou": iou,
            }
        else:
            tp, fp, tn, fn, _ = self.val_metrics["val_scores"].compute()
            miou = self.val_metrics["val_miou"].compute()
            val_dict = {
                "val_f1": f1,
                "val_tp": tp.float(),
                "val_fp": fp.float(),
                "val_tn": tn.float(),
                "val_fn": fn.float(),
                "val_iou": tp.float() / (tp + fp + fn).float(),
                "val_mean_iou": miou,
            }

        self.log_dict(val_dict, on_step=False, on_epoch=True)
        self.val_metrics.reset()
