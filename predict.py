import os
import sys
from typing import Optional

import albumentations as A
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.augmentation import AugParams, get_aug_list

sys.path.append("../")
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from src.data.augmentation import alb_transform_wrapper
from src.data.data_loader import SegDataModule
from src.lightning_modules.custom_module import Segmentation
from src.models.utils import handle_outputs, model_getter
from src.utils.crf_utils import refine_mask_with_crf
from src.utils.io import find_by_basename, read_image, read_mask
from src.utils.misc import reduce_precision, register_resolvers
from src.utils.seg_losses import loss_getter

register_resolvers()
reduce_precision()

class TestImageOnlyDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        height: int,
        width: int,
        image_exts=("png", "jpg", "jpeg"),
        transform=None,  # <-- this will be alb_transform_wrapper(val_tf)
    ):
        super().__init__()
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
        self.image_dir = image_dir
        self.h, self.w = height, width
        self.image_exts = image_exts
        self.transform = transform

        self._basenames = []
        for fname in sorted(os.listdir(image_dir)):
            stem, _ = os.path.splitext(fname)
            if find_by_basename(image_dir, stem, image_exts):
                self._basenames.append(stem)
        if not self._basenames:
            raise RuntimeError(f"No images found in {image_dir} with extensions {image_exts}.")

    def __len__(self):
        return len(self._basenames)

    def __getitem__(self, idx):
        stem = self._basenames[idx]
        img_path = find_by_basename(self.image_dir, stem, self.image_exts)
        if img_path is None:
            raise FileNotFoundError(
                f"No image found for '{stem}' in {self.image_dir} with exts {self.image_exts}."
            )

        image = read_image(img_path)  # same helper you use in train/val

        if self.transform is not None:
            # Mirror your train/val wrapper exactly: supply a dummy mask
            dummy_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image_t, _ = self.transform(image, dummy_mask)  # returns tensors
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image_t, os.path.basename(img_path)

def _collate_names_only(batch):
    imgs, names = zip(*batch)
    return default_collate(imgs), list(names)

def _get_test_dataloader(cfg: DictConfig):
    val_tf = A.Compose([A.Resize(cfg.dataset.height, cfg.dataset.width), ToTensorV2()])
    # Use your wrapper, just like SegDataModule does
    wrapped = alb_transform_wrapper(val_tf)

    ds = TestImageOnlyDataset(
        image_dir=cfg.predict.test_image_dir,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        transform=wrapped,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=lambda batch: (default_collate([b[0] for b in batch]),
                                  [b[1] for b in batch]),
    )

@hydra.main(version_base=None, config_path="config", config_name="pl_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    assert cfg.dataset.num_labels >= 2, (
        "num_labels must be at least 2 in order for softmax to work. If you have binary masks, "
        "set num_labels=2."
    )

    stage = "val"
    do_augment = False
    only_metrics = False
    save_only_pos = True
    binarize_masks = True
    top_classes = -1  # Use -1 to skip this step
    use_crf = True

    if cfg.get("predict") is not None:
        gt_folder = cfg.predict.get("gt_folder", None)
        allow_mask_skip = cfg.predict.get("allow_mask_skip", False)
    else:
        gt_folder = None

    if stage == "test" and not gt_folder:
        raise ValueError("For stage='test', cfg.predict.gt_folder is required.")

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
    if stage == "test":
        dataloader = _get_test_dataloader(cfg)
    else:
        dataloader = _get_dataloader(cfg, stage, do_augment=do_augment)

    if not only_metrics:
        output_dir = "out"
        os.makedirs(output_dir, exist_ok=True)

    all_pred = []
    all_target = []
    total_loss = 0.0
    batch_count = 0

    tp = tn = fp = fn = 0

    def load_gt_mask(mask_folder, img_name) -> Optional[np.ndarray]:
        stem = os.path.splitext(img_name)[0]
        mask_path = find_by_basename(mask_folder, stem, exts=["png", "jpg", "jpeg"])
        if mask_path is None:
            if allow_mask_skip:
                return None
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        m = read_mask(mask_path)

        # --- Normalize binary masks: 0/255 -> 0/1 (and any positive -> 1) ---
        if cfg.dataset.name == "cancer":
            # Ensure integer type and map strictly to {0,1}
            m = (m.astype(np.int64) > 0).astype(np.int64)

        return m

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Doing inference...")):
            if stage == "test":
                x, x_names = batch
                y = None
            else:
                x, y, x_names = batch
            x = x.to("cuda")
            if y is not None:
                y = y.to("cuda")
            outputs = model(x)
            outputs = handle_outputs(outputs, x, model.nametag)

            # --- Probabilities for CRF (B, C, H, W) ---
            if use_crf:
                probs_for_crf = torch.softmax(outputs, dim=1)  # (B,C,H,W)

            if y is not None:  # train/val only
                loss = loss_fn(outputs, y.unsqueeze(1).long())
                total_loss += loss.item()
                batch_count += 1

            preds = torch.argmax(outputs, dim=1).int()  # (B,H,W)
            target = y.int() if y is not None else None

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

            for i in range(preds.shape[0]):
                # Get predictions
                pr_mask = preds[i].cpu().numpy()  # original prediction without CRF

                img_name = os.path.splitext(x_names[i])[0]

                if gt_folder is not None:
                    gt_mask = load_gt_mask(gt_folder, x_names[i])
                    if gt_mask is None:
                        continue
                    gt_mask = cv2.resize(
                        gt_mask, (pr_mask.shape[1], pr_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    gt_mask = target[i].cpu().numpy()


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

                    num_colors = cfg.dataset.num_labels
                    # Apply colormap (cmap expects values normalized by the number of labels)
                    cmap = plt.get_cmap("viridis", num_colors)
                    # cmap returns RGBA, we only want RGB
                    pred_colored_original = cmap(pr_mask / float(num_colors))[:, :, :3]
                    target_colored = cmap(gt_mask / float(num_colors))[:, :, :3]

                    # Override the background (label 0) to use the original image color
                    background_mask_pred = pr_mask == 0
                    pred_colored_original[background_mask_pred] = img[background_mask_pred]

                    background_mask_target = gt_mask == 0
                    target_colored[background_mask_target] = img[background_mask_target]

                    # Create overlays by blending the original image with the colored masks
                    overlay_pred = np.clip(0.6 * img + 0.4 * pred_colored_original, 0.0, 1.0)
                    overlay_target = np.clip(0.6 * img + 0.4 * target_colored, 0.0, 1.0)

                    overlay_crf = None
                    if use_crf:
                        # Prepare inputs for CRF
                        # Use the *original image* for the bilateral term (uint8 expected; your function handles floats too)
                        img_uint = img  # function will normalize if needed

                        # Per-image class probabilities for CRF: (C,H,W) numpy
                        probs_crf_np = probs_for_crf[i].detach().cpu().numpy()  # (C,H,W)
                        num_crf_classes = probs_crf_np.shape[0]

                        # Run CRF
                        refined_mask = refine_mask_with_crf(
                            image=img_uint,
                            mask_prob=probs_crf_np,
                            num_classes=num_crf_classes,
                            iterations=10,
                            gaussian_sxy=3,
                            bilateral_sxy=80,
                            bilateral_srgb=5,
                            compat_gaussian=3,
                            compat_bilateral=10,
                        )  # (H,W) int labels in [0..C-1]

                        # If you later merge classes / binarize for visualization, mirror that logic
                        refined_vis = refined_mask.copy()
                        if top_classes > 0:
                            refined_vis = np.where(
                                refined_vis >= cfg.dataset.num_labels - top_classes, refined_vis, 0
                            )
                        if binarize_masks:
                            refined_vis = (refined_vis > 0).astype(np.int32)

                        num_colors = cfg.dataset.num_labels
                        cmap = plt.get_cmap("viridis", num_colors)
                        crf_colored = cmap(refined_vis / float(num_colors))[:, :, :3]
                        background_mask_crf = refined_vis == 0
                        crf_colored[background_mask_crf] = img[background_mask_crf]
                        overlay_crf = np.clip(0.6 * img + 0.4 * crf_colored, 0.0, 1.0)

                    img_name = x_names[i].removesuffix(".png")

                    # --- PLOT COMPARISON --- #
                    panels = [
                        ("Original", img),
                        ("Prediction", overlay_pred),
                        ("GT", overlay_target),
                    ]
                    if use_crf and overlay_crf is not None:
                        panels.insert(2, ("Pred + CRF", overlay_crf))

                    plt.figure(figsize=(18 if use_crf else 14, 4))
                    for idx, (title, im) in enumerate(panels):
                        plt.subplot(1, len(panels), idx + 1)
                        plt.imshow(im)
                        plt.title(title)
                        plt.axis("off")

                    plt.savefig(
                        os.path.join(
                            output_dir, f"{img_name}_comparison{'_crf' if use_crf else ''}.png"
                        ),
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

                    if use_crf and overlay_crf is not None:
                        plt.imsave(
                            os.path.join(additional_dir, f"{img_name}_prediction_crf.png"),
                            overlay_crf,
                        )

        # Compute overall metrics
        all_pred = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)
        avg_loss = (total_loss / batch_count) if batch_count > 0 else 0.0
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
    
    if cfg.dataset.get("aug_params") is not None:
        aug_list = get_aug_list(AugParams(**cfg.dataset.aug_params)) if do_augment else None
    else:
        aug_list = None
    data_module = SegDataModule(
        image_dir=cfg.dataset.image_dir,
        mask_dir=cfg.dataset.mask_dir,
        batch_size=cfg.batch_size,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        augment=cfg.dataset.augment if do_augment is None else do_augment,
        aug_list=aug_list,
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
