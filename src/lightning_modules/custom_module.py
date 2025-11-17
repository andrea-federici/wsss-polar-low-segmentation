import os
from typing import Mapping, Optional, Type

import cv2
import numpy as np
import torch
import torchmetrics
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torchmetrics.classification import (MulticlassF1Score,
                                         MulticlassJaccardIndex,
                                         MulticlassStatScores)

PRINT_EVERY = 100  # change as you like

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

        # ---- update dynamic-bootstrapping progress ----
        if hasattr(self.loss, "update_progress") and self.trainer is not None:
            est_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            max_epochs = getattr(self.trainer, "max_epochs", None)
            max_steps = getattr(self.trainer, "max_steps", None)

            total_iterations = None
            if max_steps is not None and max_steps != -1:
                total_iterations = max(1, int(max_steps))
            elif est_steps is not None and max_epochs is not None and est_steps > 0:
                total_iterations = max(1, int(est_steps * max_epochs))

            if total_iterations is not None:
                progress = min(1.0, float(self.global_step + 1) / float(total_iterations))
                self.loss.update_progress(progress)

        # ---- compute loss ----
        loss = self.loss(y_pred, y.unsqueeze(1).long())
        self.train_metrics.update(y_pred.argmax(1).detach().int(), y.int())

        # ---- diagnostics: CE/Dice/λ stats/β ----
        # if (self.global_step % PRINT_EVERY == 0) or (batch_idx == 0):
        #     with torch.no_grad():
        #         # current progress & bootstrap factor
        #         progress = getattr(self.loss, "progress", float("nan"))
        #         beta = self.loss._current_bootstrap_factor() if hasattr(self.loss, "_current_bootstrap_factor") else float("nan")

        #         # rebuild the same tensors used inside the loss to get consistent diagnostics
        #         target, one_hot = self.loss._prepare_target(y.unsqueeze(1).long())
        #         probs = torch.softmax(y_pred, dim=1)
        #         log_probs = torch.log_softmax(y_pred, dim=1)

        #         class_trust = self.loss.class_trust.to(device=y_pred.device)
        #         pixel_trust = class_trust.gather(0, target.view(-1)).view_as(target)
        #         lam = (1.0 - pixel_trust) * beta
        #         lam = lam.clamp(0.0, 1.0).unsqueeze(1)

        #         mixed_target = (1.0 - lam) * one_hot.to(device=y_pred.device) + lam * probs.detach()
        #         ce = -(mixed_target * log_probs).sum(dim=1).mean()

        #         # Dice (mirrors your loss)
        #         dims = (0,) + tuple(range(2, y_pred.dim()))
        #         intersection = torch.sum(probs * one_hot.to(device=y_pred.device), dim=dims)
        #         denominator = torch.sum(probs + one_hot.to(device=y_pred.device), dim=dims)
        #         dice_per_class = 1.0 - (2.0 * intersection + self.loss.smooth) / (denominator + self.loss.smooth)

        #         if self.loss.exclude_background_from_dice:
        #             dice_per_class = dice_per_class[1:]
        #             trust_weights = class_trust[1:]
        #         else:
        #             trust_weights = class_trust

        #         weight_sum = torch.clamp(trust_weights.sum(), min=self.loss.smooth)
        #         dice = (dice_per_class * trust_weights).sum() / weight_sum

        #         lam_mean = lam.mean().item()
        #         lam_min = lam.min().item()
        #         lam_max = lam.max().item()

        #     rank_zero_info(
        #         f"[epoch={self.current_epoch} step={self.global_step}] "
        #         f"prog={progress:.3f} β={beta:.3f} "
        #         f"λ(mean/min/max)={lam_mean:.3f}/{lam_min:.3f}/{lam_max:.3f} "
        #         f"CE={ce.item():.4f} Dice={dice.item():.4f} "
        #         f"Total={loss.item():.4f}"
        #     )

        #     # Optional: also log these scalars to your logger
        #     self.log("bootstrapping/beta", torch.tensor(beta, device=loss.device), on_step=True, prog_bar=False, sync_dist=self.sync_dist)
        #     self.log("bootstrapping/lambda_mean", torch.tensor(lam_mean, device=loss.device), on_step=True, prog_bar=False, sync_dist=self.sync_dist)
        #     self.log("loss_components/ce", ce, on_step=True, prog_bar=False, sync_dist=self.sync_dist)
        #     self.log("loss_components/dice", dice, on_step=True, prog_bar=False, sync_dist=self.sync_dist)

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
