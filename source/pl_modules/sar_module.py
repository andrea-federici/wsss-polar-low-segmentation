from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassStatScores
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import NeptuneLogger
from lightning.pytorch.utilities import grad_norm

from source import utils
from .base_module import BaseModule


class Segmentation(BaseModule):

    def maybe_log_preds(self, batch, y_pred, batch_idx, title):
        if self.plot_preds_at_epoch is not None:
            b_idx = self.plot_preds_at_epoch.get("batch", 0)
            s_idx = self.plot_preds_at_epoch.get("samples", 1)
            every = self.plot_preds_at_epoch.get("every", 1)
            if batch_idx == b_idx and self.current_epoch % every == 0:
                x, y = batch
                img = utils.img_logging.xy_grid(x[:s_idx], y[:s_idx], y_pred[:s_idx])
                title = f"{title}@ep{self.current_epoch}"
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_image(title, img, self.current_epoch)
                elif isinstance(self.logger, NeptuneLogger):
                    img = img.permute(1, 2, 0)
                    self.logger.log_tensor_img(img, title)
                else:
                    raise TypeError(f"Logger type not OK: {type(self.logger)}")

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        num_labels: int,
        loss_fn: Type,
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
        self.loss_fn = loss_fn
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
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val_f1": MulticlassF1Score(num_classes=num_labels, average="macro"),
                "val_scores": MulticlassStatScores(
                    num_classes=num_labels, average="macro"
                ),
            }
        )

    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y.long())
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.train_metrics.update(y_pred.argmax(1).detach().int(), y.int())

        # log preds
        if "train" in self.plot_preds_at_epoch["set"]:
            self.maybe_log_preds(batch, y_pred, batch_idx, "train")

        return {"loss": loss}

    # - - - - Validation - - - - - - - - - - - - - - - - - - - - #

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y.long())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        self.val_metrics.update(y_pred.argmax(1).int(), y.int())

        # log preds
        if "val" in self.plot_preds_at_epoch["set"]:
            self.maybe_log_preds(batch, y_pred, batch_idx, "val")

        return {"loss": loss}
