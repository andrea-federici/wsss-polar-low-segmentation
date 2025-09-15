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
            self.register_buffer(
                "prob_map", torch.tensor([0.0, 1.0, 0.7, 0.15], dtype=torch.float32)
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

    def calc_metrics_detailed(
        self,
        all_pred: np.ndarray,
        all_target: np.ndarray,
        avg_loss: float,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        num_labels: int,
        verbose: bool = True,
    ) -> dict:
        """Return detailed per-class + per-image metrics. Expects integer labels 0..num_labels-1."""
        if verbose:
            print("Calculating detailed metrics...")

        all_pred = np.asarray(all_pred).ravel()
        all_target = np.asarray(all_target).ravel()

        if all_pred.shape != all_target.shape:
            raise ValueError("all_pred and all_target must have the same shape.")

        results = {}
        # Basic global metrics (safe wrappers)
        try:
            mean_iou = jaccard_score(
                all_target, all_pred, average="macro", labels=list(range(num_labels))
            )
        except Exception:
            # fallback to manual per-class IoU mean (ignore classes with no union)
            ious = []
            for k in range(num_labels):
                pred_k = all_pred == k
                targ_k = all_target == k
                inter = np.logical_and(pred_k, targ_k).sum()
                union = np.logical_or(pred_k, targ_k).sum()
                if union == 0:
                    ious.append(np.nan)
                else:
                    ious.append(inter / union)
            mean_iou = float(np.nanmean(ious))

        results["mean_iou"] = float(mean_iou)
        results["avg_loss"] = float(avg_loss)
        results.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})

        # Per-class counters and metrics
        per_class = {}
        for k in range(num_labels):
            pred_k = all_pred == k
            targ_k = all_target == k

            tp_k = int(np.logical_and(pred_k, targ_k).sum())
            fp_k = int(np.logical_and(pred_k, np.logical_not(targ_k)).sum())
            fn_k = int(np.logical_and(np.logical_not(pred_k), targ_k).sum())
            tn_k = int(np.logical_and(np.logical_not(pred_k), np.logical_not(targ_k)).sum())

            # Precision / Recall / F1 (handle zero division)
            prec_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
            rec_k = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
            f1_k = (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

            # Dice and IoU
            denom = pred_k.sum() + targ_k.sum()
            dice_k = (2.0 * tp_k / denom) if denom > 0 else np.nan
            union = np.logical_or(pred_k, targ_k).sum()
            iou_k = (tp_k / union) if union > 0 else np.nan

            per_class[k] = {
                "tp": tp_k,
                "fp": fp_k,
                "fn": fn_k,
                "tn": tn_k,
                "precision": float(prec_k),
                "recall": float(rec_k),
                "f1": float(f1_k),
                "dice": (float(dice_k) if not np.isnan(dice_k) else None),
                "iou": (float(iou_k) if not np.isnan(iou_k) else None),
                "support": int(targ_k.sum()),
                "pred_count": int(pred_k.sum()),
            }

        results["per_class"] = per_class

        # Macro / Micro F1 using sklearn (safe)
        try:
            results["macro_f1"] = float(
                f1_score(all_target, all_pred, average="macro", zero_division=0)
            )
            results["micro_f1"] = float(
                f1_score(all_target, all_pred, average="micro", zero_division=0)
            )
        except Exception:
            results["macro_f1"] = None
            results["micro_f1"] = None

        # Macro dice (mean of per-class dice ignoring None)
        dice_vals = [v["dice"] for v in per_class.values() if v["dice"] is not None]
        results["macro_dice"] = float(np.nanmean(dice_vals)) if len(dice_vals) else 0.0

        # Per-image IoU/Dice for minority class: give distribution (useful to find whether failures are global or a few bad images)
        # This requires having predictions/targets per image. We only have flattened arrays here,
        # so to compute per-image metrics you should pass a list/array of per-image preds/targets instead.
        # We'll provide a helper below; here we compute overall per-image-like approximations only if shapes are provided as (N, H, W).
        # So we return an empty list for per_image unless the caller supplied shaped arrays (not flattened).
        results["per_image"] = {}

        if verbose:
            print("--- Summary ---")
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            print(f"Avg Loss: {results['avg_loss']:.6f}")
            print(f"Mean IoU (macro): {results['mean_iou']:.6f}")
            print(f"Macro Dice: {results['macro_dice']:.6f}")
            print(f"Macro F1: {results['macro_f1']:.6f}, Micro F1: {results['micro_f1']:.6f}")
            for k, d in per_class.items():
                dice_str = "nan" if d["dice"] is None else f"{d['dice']:.4f}"
                print(
                    f"Class {k}: support={d['support']}, pred_count={d['pred_count']}, TP={d['tp']}, FP={d['fp']}, FN={d['fn']}, Dice={dice_str}, F1={d['f1']:.4f}"
                )

        # If user provided shaped arrays, compute per-image IoU for minority_class (optional separate helper recommended)
        return results

    def validation_step(self, batch, batch_idx):
        x, y, x_names = batch
        y_pred = self.forward(x)
        B, C = y_pred.shape[:2]

        if self.num_labels == 2:
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

        gt_mask_folder = "data/bus/gt_masks"
        gt_masks = []
        for fname in x_names:
            gt_mask = self.load_gt_mask(gt_mask_folder, fname)
            gt_mask = cv2.resize(
                gt_mask,
                (512, 512),
                interpolation=cv2.INTER_NEAREST,
            )
            gt_mask = torch.tensor(gt_mask, dtype=torch.long, device=y_pred.device)
            gt_masks.append(gt_mask)
        gt_masks = torch.stack(gt_masks)

        if C == 1:
            prob = torch.sigmoid(y_pred)
            preds_bin = (prob >= 0.5).int().squeeze(1)
        else:
            preds = y_pred.argmax(1).int()
            preds_bin = (preds > 0).int()

        gt_bin = (gt_masks > 0).int()

        # Flatten for metrics
        preds_np = preds_bin.cpu().numpy().ravel()
        gts_np = gt_bin.cpu().numpy().ravel()

        # Confusion counts
        tp = int(((preds_bin == 1) & (gt_bin == 1)).sum().item())
        fp = int(((preds_bin == 1) & (gt_bin == 0)).sum().item())
        fn = int(((preds_bin == 0) & (gt_bin == 1)).sum().item())
        tn = int(((preds_bin == 0) & (gt_bin == 0)).sum().item())

        # Compute detailed metrics
        metrics = self.calc_metrics_detailed(
            all_pred=preds_np,
            all_target=gts_np,
            avg_loss=loss.item(),
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            num_labels=2,  # binary after collapsing
            verbose=False,
        )

        # Log key class 1 metrics
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.sync_dist
        )
        self.log("val_dice_class1", metrics["per_class"][1]["dice"], on_epoch=True, prog_bar=True)
        self.log("val_iou_class1", metrics["per_class"][1]["iou"], on_epoch=True, prog_bar=True)
        self.log("val_f1_class1", metrics["per_class"][1]["f1"], on_epoch=True)

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
