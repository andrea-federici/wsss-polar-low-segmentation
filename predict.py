import os
import sys
from typing import Optional

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.append("../")
from sklearn.metrics import f1_score, jaccard_score

from src.data.data_loader import DataModuleWrapper
from src.lightning_modules.custom_module import Segmentation
from src.models.utils import handle_outputs, model_getter
from src.utils.constants import SUPPORTED_DATASETS
from src.utils.misc import reduce_precision, register_resolvers
from src.utils.seg_losses import SoftDiceBCELoss, loss_getter

register_resolvers()
reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="pl_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    stage = "val"
    do_augment = False
    only_metrics = True
    save_only_pos = False
    binarize_masks = True
    top_classes = 1  # Use -1 to skip this step

    gt_folder = cfg.predict.get("gt_folder") if cfg.get("predict") is not None else None

    # Ensure that the dataset specified is supported
    if cfg.dataset.name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. "
            f"Supported datasets are: {', '.join(SUPPORTED_DATASETS)}"
        )

    # Ensure that the checkpoint path was specified
    if not cfg.checkpoint.path:
        raise ValueError(
            "No checkpoint was specified. Please specify the "
            "checkpoint path in the config file or from the command line "
            "when running the script."
        )

    checkpoint_path = os.path.join(cfg.checkpoint.base_folder, cfg.checkpoint.path)

    # Ensure that the checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    # Model
    model = model_getter(cfg.model.name, cfg, print_summary=False)

    # Loss
    loss_fn = loss_getter(name=cfg.loss.name, **cfg.loss.hparams)

    # Optim scheduler
    if cfg.get("lr_scheduler") is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    seg = Segmentation.load_from_checkpoint(
        checkpoint_path,
        model=model,
        num_labels=cfg.dataset.num_labels,
        loss=loss_fn,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        log_lr=cfg.log_lr,
        log_grad_norm=cfg.log_grad_norm,
        plot_dict=dict(cfg.plot_preds_at_epoch),
        map_location="cuda:0",  # TODO: make this configurable.
    )

    model = seg.model
    model.eval().to("cuda")

    # Data module:
    dataloader = _get_dataloader(cfg, stage, do_augment=do_augment)

    if not only_metrics:
        output_dir = "out"
        os.makedirs(output_dir, exist_ok=True)

    all_pred = []
    all_target = []
    total_loss = 0.0
    batch_count = 0

    tp = tn = fp = fn = 0

    def load_gt_mask(mask_folder, img_name):
        mask_path = os.path.join(mask_folder, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = (mask == 255).astype(np.uint8)
        return mask

    prob_map = torch.tensor([0.0, 0.95, 0.75, 0.3], dtype=torch.float32)
    prob_map_np = prob_map.numpy()

    with torch.no_grad():
        for batch_idx, (x, y, x_names) in enumerate(tqdm(dataloader, desc="Doing inference...")):
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            outputs = handle_outputs(outputs, x, model.nametag)

            # ---- construct targets for loss ----
            # If loss expects soft targets (binary BCE+soft-dice wrapper), build float soft targets
            if isinstance(
                loss_fn, SoftDiceBCELoss
            ):  # or check a flag: getattr(loss_fn,'expects_soft',False)
                # ensure y is shape (B,H,W) for indexing
                if y.ndim == 4 and y.shape[1] == 1:
                    y_index = y.squeeze(1)
                else:
                    y_index = y

                # y_index must be integer indices 0..K
                y_index = y_index.long()

                # map labels -> soft probabilities and add channel dim (B,1,H,W)
                soft_targets = prob_map.to(outputs.device)[y_index]  # shape (B,H,W)
                soft_targets = soft_targets.unsqueeze(1).float()  # shape (B,1,H,W)

                print("outputs.shape:", outputs.shape)  # expect (B,1,H,W)
                print("soft_targets.shape:", soft_targets.shape)  # currently (B,1,H,W,4)
                print("prob_map.shape:", prob_map.shape)  # is this (5,) or (5,4) ?
                print("soft_targets dtype:", soft_targets.dtype)

                # compute loss (loss_fn expects logits + soft_targets floats)
                loss = loss_fn(outputs, soft_targets)

            else:
                # multiclass losses expect integer labels or one-hot depending on your loss.
                # Keep old behaviour but make sure shapes are correct
                # Many multiclass losses expect shape (B, H, W) long labels, or (B, C, H, W) one-hot.
                loss = loss_fn(outputs, y.unsqueeze(1).long())

            # ---- produce preds for downstream metrics and optional merging ----
            if outputs.ndim == 4 and outputs.shape[1] == 1:
                # binary output: use sigmoid and threshold
                probs = torch.sigmoid(outputs)  # (B,1,H,W)
                probs_np = probs.squeeze(1).cpu().numpy()  # (B,H,W), float in [0,1]

                # Map each pixel's probability to the closest entry in prob_map (vectorized)
                # prob_map_np is defined earlier as numpy array length L (labels 0..L-1)
                # class_map_np: (B,H,W) with integer labels 0..L-1
                # distance across labels: |p - prob_map[label]|
                # NOTE: this handles background (label 0) naturally if prob_map[0] == 0.0
                d = np.abs(probs_np[..., None] - prob_map_np[None, None, None, :])  # (B,H,W,L)
                class_map_np = d.argmin(axis=-1).astype(np.int32)  # (B,H,W)

                # Set preds tensor (for metric computations that expect tensor)
                preds = (
                    torch.from_numpy(class_map_np).to(outputs.device).int()
                )  # (B,H,W) int on device
            else:
                # multiclass output: use argmax
                preds = torch.argmax(outputs, dim=1).int()  # (B,H,W)

            target = y.int().squeeze(1) if (y.ndim == 4 and y.shape[1] == 1) else y.int()

            # Merge all labels < top_classes into background
            if top_classes > 0:
                preds = torch.where(
                    preds >= cfg.dataset.num_labels - top_classes,
                    preds,
                    torch.zeros_like(preds),
                )

            # Convert to binary: background (0) vs. foreground (1+)
            if binarize_masks:
                preds = (preds > 0).int()

            total_loss += loss.item()
            batch_count += 1

            for i in range(preds.shape[0]):
                # Get predictions
                pr_mask = preds[i].cpu().numpy()  # original prediction without CRF

                if gt_folder is not None:
                    gt_mask = load_gt_mask(gt_folder, x_names[i])
                    gt_mask = cv2.resize(
                        gt_mask,
                        (pr_mask.shape[1], pr_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    gt_mask = target[i].cpu().numpy()  # ground truth mask

                all_pred.append(pr_mask.flatten())
                all_target.append(gt_mask.flatten())

                if gt_mask.sum() == 0:
                    gt_label = 0
                else:
                    gt_label = 1

                if pr_mask.sum() == 0:
                    pred_label = 0
                else:
                    pred_label = 1

                # Update confusion counters
                if pred_label == 1 and gt_label == 1:
                    tp += 1
                elif pred_label == 0 and gt_label == 0:
                    tn += 1
                elif pred_label == 1 and gt_label == 0:
                    fp += 1
                elif pred_label == 0 and gt_label == 1:
                    fn += 1

                if save_only_pos and gt_label == 0:
                    continue

                if not only_metrics:
                    # Convert image from tensor (CHW) to numpy (HWC) and normalize
                    img = x[i].cpu().permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min())

                    # Resize masks to match original image size
                    w, h = img.shape[1], img.shape[0]
                    pr_mask = cv2.resize(
                        pr_mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    gt_mask = cv2.resize(
                        gt_mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    if isinstance(loss_fn, SoftDiceBCELoss):
                        num_colors = prob_map_np.shape[0]
                    else:
                        num_colors = cfg.dataset.num_labels

                    # Apply colormap (cmap expects values normalized by the number of labels)
                    cmap = plt.get_cmap("viridis", num_colors)
                    # cmap returns RGBA, we only want RGB
                    pred_colored_original = cmap(pr_mask / float(num_colors))[:, :, :3]
                    target_colored = cmap(gt_mask / float(num_colors))[:, :, :3]

                    # print(f"File name: {x_names[i]}")
                    # print("Shapes:")
                    # print(f"  pr_mask: {pr_mask.shape}")
                    # print(f"  gt_mask: {gt_mask.shape}")
                    # print(f"  img: {img.shape}")
                    # print(f"  pred_colored_original: {pred_colored_original.shape}")
                    # print(f"  target_colored: {target_colored.shape}")

                    # Override the background (label 0) to use the original image color
                    background_mask_pred = pr_mask == 0
                    pred_colored_original[background_mask_pred] = img[background_mask_pred]

                    background_mask_target = gt_mask == 0
                    target_colored[background_mask_target] = img[background_mask_target]

                    # Create overlays by blending the original image with the colored masks
                    overlay_pred = np.clip(0.6 * img + 0.4 * pred_colored_original, 0.0, 1.0)
                    overlay_target = np.clip(0.6 * img + 0.4 * target_colored, 0.0, 1.0)

                    img_name = x_names[i].removesuffix(".png")

                    ### --- PLOT COMPARISON --- ###
                    plt.figure(figsize=(14, 4))
                    for idx, (im, title) in enumerate(
                        zip([img, overlay_pred, overlay_target], ["Original", "Prediction", "GT"])
                    ):
                        plt.subplot(1, 3, idx + 1)
                        plt.imshow(im)
                        plt.title(title)
                        plt.axis("off")
                    plt.savefig(
                        os.path.join(output_dir, f"{img_name}_comparison.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                    additional_dir = os.path.join(output_dir, "additional")
                    os.makedirs(additional_dir, exist_ok=True)

                    plt.imsave(os.path.join(additional_dir, f"{img_name}_original.png"), img)
                    plt.imsave(
                        os.path.join(additional_dir, f"{img_name}_prediction.png"),
                        overlay_pred,
                    )
                    plt.imsave(
                        os.path.join(additional_dir, f"{img_name}_pseudomask.png"),
                        overlay_target,
                    )

        # Compute overall metrics
        all_pred = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)
        avg_loss = total_loss / batch_count if batch_count else 0.0
        calc_metrics_detailed(
            all_pred,
            all_target,
            avg_loss,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            num_labels=cfg.dataset.num_labels if not binarize_masks else 2,
            verbose=True,
        )


def calc_metrics_detailed(
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


# TODO: do we want to keep this option to override the do_augment flag? Or do we just
# want to use the one from the config file?
def _get_dataloader(cfg: DictConfig, stage: str, do_augment: Optional[bool] = None):
    supported_stages = ["train", "val", "test"]
    if stage not in supported_stages:
        raise ValueError(
            f"Stage {stage} not supported. Supported stages: {', '.join(supported_stages)}"
        )
    if cfg.dataset.name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. "
            f"Supported datasets are: {', '.join(SUPPORTED_DATASETS)}"
        )
    data_module = DataModuleWrapper(
        image_dir=cfg.dataset.image_dir,
        mask_dir=cfg.dataset.mask_dir,
        batch_size=cfg.batch_size,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        augment=cfg.dataset.augment if do_augment is None else do_augment,
        num_workers=cfg.workers,
    )
    data_module.prepare_data()
    data_module.setup(stage=stage)
    if stage == "train":
        return data_module.train_dataloader()
    elif stage == "val":
        return data_module.val_dataloader()
    else:
        return data_module.test_dataloader()


if __name__ == "__main__":
    run()
