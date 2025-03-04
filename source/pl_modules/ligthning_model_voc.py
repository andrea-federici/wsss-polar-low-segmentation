from typing import Optional, Mapping, Type
import torch
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassStatScores
import lightning
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import NeptuneLogger
from lightning.pytorch.utilities import grad_norm
from source import utils

class Segmentation(lightning.LightningModule):

    def maybe_log_preds(self, batch, y_pred, batch_idx, title):
        if self.plot_preds_at_epoch is not None:
            b_idx = self.plot_preds_at_epoch.get('batch', 0)
            s_idx = self.plot_preds_at_epoch.get('samples', 1)
            every = self.plot_preds_at_epoch.get('every', 1)
            if batch_idx == b_idx and self.current_epoch%every==0:
                x,y = batch
                img = utils.img_logging.xy_grid(x[:s_idx], y[:s_idx], y_pred[:s_idx])
                title = f'{title}@ep{self.current_epoch}'
                if isinstance(self.logger, TensorBoardLogger):
                    self.logger.experiment.add_image(title, img, self.current_epoch)
                elif isinstance(self.logger, NeptuneLogger):
                    img = img.permute(1,2,0)
                    self.logger.log_tensor_img(img, title)
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

    def forward(self, x):
        outputs = self.model(x)
        if self.model.nametag == '___segformer':
            outputs = torch.nn.functional.interpolate(outputs['logits'], 
                                                      size=x.shape[-2:], 
                                                      mode='bilinear', 
                                                      align_corners=False)
        if self.model.nametag in ['___dpt', '___uper']:
            outputs = outputs['logits']
        return outputs
   
    def configure_optimizers(self):
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer,
                                             **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg
    
    def on_before_optimizer_step(self, optimizer):
        if self.log_grad_norm:
            # inspect (unscaled) gradients here
            self.log_dict(grad_norm(self, norm_type=2))
    
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
    
    def on_train_epoch_end(self):
        f1 = self.train_metrics['train_f1'].compute()
        tp, fp, tn, fn, _ = self.train_metrics['train_scores'].compute()
        train_dict = {'train_f1': f1, 'train_tp': tp.float(), 'train_fp': fp.float(), 
                      'train_tn': tn.float(), 'train_fn': fn.float(),
                      'train_iou': tp.float() / (tp + fp + fn).float()}
        self.log_dict(train_dict, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_train_epoch_start(self) -> None:
        if self.log_lr:
            # Log learning rate
            optimizers = self.optimizers()
            if isinstance(optimizers, list):
                for i, optimizer in enumerate(optimizers):
                    lr = optimizer.optimizer.param_groups[0]['lr']
                    self.log(f'lr_{i}', lr, on_step=False, on_epoch=True,
                             logger=True, prog_bar=False, batch_size=1, sync_dist=self.sync_dist)
            else:
                lr = optimizers.optimizer.param_groups[0]['lr']
                self.log(f'lr', lr, on_step=False, on_epoch=True,
                         logger=True, prog_bar=False, batch_size=1, sync_dist=self.sync_dist)
    
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
    
    def on_validation_epoch_end(self):
        f1 = self.val_metrics['val_f1'].compute()
        tp, fp, tn, fn, _ = self.val_metrics['val_scores'].compute()
        val_dict = {'val_f1': f1, 'val_tp': tp.float(), 'val_fp': fp.float(), 
                    'val_tn': tn.float(), 'val_fn': fn.float(),
                    'val_iou': tp.float() / (tp + fp + fn).float()}
        self.log_dict(val_dict, on_step=False, on_epoch=True) 
        self.val_metrics.reset()