from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassStatScores
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import NeptuneLogger

from source import utils
from .base_module import BaseModule 

class Segmentation(BaseModule):

    def maybe_log_preds(self, batch, y_pred, batch_idx, title):
        if self.plot_preds_at_epoch is not None:
            b_idx = self.plot_preds_at_epoch.get('batch', 0)
            s_idx = self.plot_preds_at_epoch.get('samples', 1)
            every = self.plot_preds_at_epoch.get('every', 1)
            if batch_idx == b_idx and self.current_epoch % every == 0:
                x, y = batch
                img = utils.img_logging.xy_grid_voc(x, y, y_pred, max_samples=s_idx)
                title = f'{title}@ep{self.current_epoch}'
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_image(title, img, self.current_epoch)
                elif isinstance(self.logger, NeptuneLogger):
                    # Neptune expects an HxWxC, so permute for logging:
                    img_np = img.permute(1,2,0)  # shape [H,W,3]
                    self.logger.log_tensor_img(img_np, title)
                else:
                    raise TypeError(f'Logger type not OK: {type(self.logger)}')

    def __init__(self, 
                 model: torch.nn.Module, 
                 loss_fn: Type,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 log_lr: bool = True,
                 log_grad_norm: bool = False,
                 sync_dist: bool = False,   # if ``True``, reduces the metric across devices. Causes overhead. Use only for multi-gpu train
                 plot_dict: Optional[Mapping] = None):
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

        num_classes = 21  # for VOC (20 foreground classes + background)

        self.train_metrics = torchmetrics.MetricCollection({
            'train_f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'train_scores': MulticlassStatScores(num_classes=num_classes, average='macro'),
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'val_f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'val_scores': MulticlassStatScores(num_classes=num_classes, average='macro'),
        })

    
    # - - - - Training - - - - - - - - - - - - - - - - - - - - - #

    def training_step(self, batch, batch_idx):
        x, y = batch  # y can contain 255 for void regions
        y_pred = self.forward(x)

        # 1) Compute the loss with ignore_index=255, if your loss supports it
        #    (For a built-in PyTorch cross-entropy, you'd do `ignore_index=255`.)
        loss = self.loss_fn(y_pred, y.long())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist)

        # 2) Convert predictions & targets to shape [B,H,W] and type int
        preds = y_pred.argmax(dim=1).int()  # shape [B,H,W]
        target = y.int()                    # shape [B,H,W]

        # 3) Mask out the void pixels so they don't show up in the metrics
        valid_mask = target != 255
        preds = preds[valid_mask]
        target = target[valid_mask]

        # 4) Update the metrics
        self.train_metrics.update(preds, target)

        # 5) (Optional) log predictions if needed
        if 'train' in self.plot_preds_at_epoch['set']:
            self.maybe_log_preds(batch, y_pred, batch_idx, 'train')

        return {'loss': loss}
    
    
    # - - - - Validation - - - - - - - - - - - - - - - - - - - - #
    def validation_step(self, batch, batch_idx):
        x, y = batch            # y in [0..20 or 255]
        y_pred = self.forward(x)

        loss = self.loss_fn(y_pred, y.long())
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=self.sync_dist)

        # Convert predictions/labels to [B,H,W] with int dtype
        preds = y_pred.argmax(dim=1).int()  # shape [B,H,W]
        target = y.int()                    # shape [B,H,W]

        # Filter out all “void” pixels
        mask = target != 255
        preds = preds[mask]
        target = target[mask]

        # Now update the metrics
        self.val_metrics.update(preds, target)

        # Logging predictions
        if 'val' in self.plot_preds_at_epoch['set']:
            self.maybe_log_preds(batch, y_pred, batch_idx, 'val')

        return {'loss': loss}