from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassStatScores,
    MulticlassJaccardIndex,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import NeptuneLogger

from src import utils
from .base_module import BaseModule


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

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "train_f1": MulticlassF1Score(num_classes=num_labels, average="macro"),
                "train_scores": MulticlassStatScores(
                    num_classes=num_labels, average="macro"
                ),
                "train_miou": MulticlassJaccardIndex(
                    num_classes=num_labels, average="macro"
                ),
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val_f1": MulticlassF1Score(num_classes=num_labels, average="macro"),
                "val_scores": MulticlassStatScores(
                    num_classes=num_labels, average="macro"
                ),
                "val_miou": MulticlassJaccardIndex(
                    num_classes=num_labels, average="macro"
                ),
            }
        )

    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y.unsqueeze(1).long())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.train_metrics.update(y_pred.argmax(1).detach().int(), y.int())

        return {"loss": loss}

    # - - - - Validation - - - - - - - - - - - - - - - - - - - - #

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y.unsqueeze(1).long())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.val_metrics.update(y_pred.argmax(1).int(), y.int())

        return {"loss": loss}
