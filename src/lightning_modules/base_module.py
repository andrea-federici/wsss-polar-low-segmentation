from typing import Mapping, Optional, Type

import lightning as pl
from lightning.pytorch.utilities import grad_norm

from src import models


class BaseModule(pl.LightningModule):
    """
    Base Lightning Module class
    """

    def __init__(
        self,
        optim_class: Optional[Type] = None,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        log_lr: bool = True,
        log_grad_norm: bool = False,
        sync_dist: bool = False,  # if True, reduces the metric across devices. Causes overhead. Use only for multi-gpu train
    ):
        super().__init__()
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()
        self.log_lr = log_lr
        self.log_grad_norm = log_grad_norm
        self.sync_dist = sync_dist

        self.save_hyperparameters()

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        """
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg

    def on_before_optimizer_step(self, optimizer):
        """
        Log gradients norm
        """
        if self.log_grad_norm:
            self.log_dict(grad_norm(self, norm_type=2))

    def on_train_epoch_start(self) -> None:
        """
        Log learning rate at the start of each epoch
        """
        if self.log_lr:
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for i, optimizer in enumerate(optimizers):
                    lr = optimizer.optimizer.param_groups[0]["lr"]
                    self.log(
                        f"lr_{i}",
                        lr,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        prog_bar=False,
                        sync_dist=self.sync_dist,
                    )
            else:
                lr = optimizers.optimizer.param_groups[0]["lr"]
                self.log(
                    "lr",
                    lr,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=False,
                    sync_dist=self.sync_dist,
                )

    def forward(self, x):
        outputs = self.model(x)
        outputs = models.utils.handle_outputs(outputs, x, self.model.nametag)
        return outputs
