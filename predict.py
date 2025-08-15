import torch
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("../")
from sklearn.metrics import jaccard_score
from source import models, data, utils

utils.misc.register_resolvers()
utils.misc.reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="predict")
def run(cfg: DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    stage = "val"
    save_only_pos = True
    binarize_masks = True
    top_classes = 2

    # Ensure that the dataset specified is supported
    supported_datasets = [
        utils.constants.PL_DATASET_NAME,
    ]
    if cfg.dataset.name not in supported_datasets:
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. "
            f"Supported datasets are: {', '.join(supported_datasets)}"
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
    model = models.utils.model_getter(cfg.model.name, cfg, print_summary=False)

    # Loss
    loss_fn = utils.seg_losses.loss_getter(name=cfg.loss.name, **cfg.loss.hparams)

    # Optim scheduler
    if cfg.get("lr_scheduler") is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    seg = models.custom_module.Segmentation.load_from_checkpoint(
        checkpoint_path,
        model=model,
        num_labels=cfg.dataset.num_labels,
        loss_fn=loss_fn,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        log_lr=cfg.log_lr,
        log_grad_norm=cfg.log_grad_norm,
        plot_dict=dict(cfg.plot_preds_at_epoch),
    )

    model = seg.model
    model.eval().to("cuda")

    # Data module:
    dataloader = _get_dataloader(cfg, stage)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    all_pred = []
    all_target = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Doing inference...")):
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            outputs = models.utils.handle_outputs(outputs, x, model.nametag)

            loss = loss_fn(outputs, y.long())
            preds = torch.argmax(outputs, dim=1).int()
            target = y.int()

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

            # Convert logits or soft outputs to probability maps
            # Assuming outputs is logits: apply softmax along the class dimension
            softmax_outputs = torch.softmax(outputs, dim=1)

            # Optional: Convert probabilities to CPU numpy arrays for CRF processing
            # Process each image in the batch
            refined_preds = []
            for i in range(softmax_outputs.shape[0]):
                # Get the original image and predicted probability map
                orig_img = x[i].cpu().permute(1, 2, 0).numpy()
                orig_img = (orig_img - orig_img.min()) / (
                    orig_img.max() - orig_img.min()
                )
                prob_map = (
                    softmax_outputs[i].cpu().numpy()
                )  # shape: (num_classes, H, W)

                # Here, we assume binary segmentation for CRF.
                # For binary, prob_map[0] is background and prob_map[1] is foreground.
                refined_mask = utils.crf_utils.refine_mask_with_crf(
                    orig_img, prob_map, num_classes=cfg.dataset.num_labels, iterations=5
                )
                refined_preds.append(refined_mask)

            # Ignore void pixels for metrics only
            mask = target != 255
            preds_for_metrics = preds[mask].flatten().cpu().numpy()
            target_for_metrics = target[mask].flatten().cpu().numpy()

            # Accumulate metrics
            all_pred.append(preds[mask].cpu().numpy().flatten())
            all_target.append(target[mask].cpu().numpy().flatten())
            total_loss += loss.item()
            batch_count += 1

            # TODO: use color map from misc.img_logging
            cmap = plt.get_cmap("viridis", cfg.dataset.num_labels)  # VOC -> 21 classes

            for i in range(preds.shape[0]):
                # Convert image from tensor (CHW) to numpy (HWC) and normalize
                img = x[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())

                # Get predictions
                pr_mask = preds[i].cpu().numpy()  # original prediction without CRF
                pr_mask_crf = refined_preds[i]  # CRF refined mask
                gt_mask = target[i].cpu().numpy()  # ground truth mask

                if gt_mask.sum() == 0:
                    gt_label = 0
                else:
                    gt_label = 1

                if save_only_pos and gt_label == 0:
                    continue

                # Resize masks to match original image size
                w, h = img.shape[1], img.shape[0]
                pr_mask = cv2.resize(
                    pr_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                pr_mask_crf = cv2.resize(
                    pr_mask_crf,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                gt_mask = cv2.resize(
                    gt_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Apply colormap (cmap expects values normalized by the number of labels)
                pred_colored_original = cmap(pr_mask / cfg.dataset.num_labels)[:, :, :3]
                pred_colored_crf = cmap(pr_mask_crf / cfg.dataset.num_labels)[:, :, :3]
                target_colored = cmap(gt_mask / cfg.dataset.num_labels)[:, :, :3]

                # Override the background (label 0) to use the original image color
                background_mask_pred = pr_mask == 0
                pred_colored_original[background_mask_pred] = img[background_mask_pred]

                background_mask_crf = pr_mask_crf == 0
                pred_colored_crf[background_mask_crf] = img[background_mask_crf]

                background_mask_target = gt_mask == 0
                target_colored[background_mask_target] = img[background_mask_target]

                # Create overlays by blending the original image with the colored masks
                overlay_original = np.clip(
                    0.6 * img + 0.4 * pred_colored_original, 0.0, 1.0
                )
                overlay_crf = np.clip(0.6 * img + 0.4 * pred_colored_crf, 0.0, 1.0)
                overlay_target = np.clip(0.6 * img + 0.4 * target_colored, 0.0, 1.0)

                # Create a three-panel figure
                plt.figure(figsize=(14, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title(f"Original Image. GT Label: {gt_label}")
                plt.axis("off")

                # Plot original prediction (without CRF)
                plt.subplot(1, 3, 2)
                plt.imshow(overlay_original)
                plt.title("Prediction (no post-processing)")
                plt.axis("off")

                # # Plot CRF refined prediction
                # plt.subplot(1, 4, 3)
                # plt.imshow(overlay_crf)
                # plt.title("CRF Refined Prediction")
                # plt.axis("off")

                # Plot ground truth mask
                plt.subplot(1, 3, 3)
                plt.imshow(overlay_target)
                plt.title("Pseudo Label")
                plt.axis("off")

                # Save and close the figure
                plt.savefig(
                    os.path.join(
                        output_dir, f"segmentation_comparison_{batch_idx}_{i}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        # Compute overall metrics
        all_pred = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)
        mean_iou = jaccard_score(all_target, all_pred, average="macro")
        avg_loss = total_loss / batch_count if batch_count else 0.0

        print(f"Final Results - Avg Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}")


def _get_dataloader(cfg: DictConfig, stage: str):
    supported_stages = ["train", "val", "test"]
    assert (
        stage in supported_stages
    ), f"Stage {stage} not supported. Supported stages: {', '.join(supported_stages)}"
    if cfg.dataset.name == utils.constants.VOC_DATASET_NAME:
        data_module = data.data_loaders.VOCDataModule(
            data_dir="data",
            batch_size=cfg.batch_size,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            num_workers=cfg.workers,
        )
    elif cfg.dataset.name == utils.constants.PL_DATASET_NAME:
        data_module = data.data_loader.DataModuleWrapper(
            image_dir=cfg.dataset.image_dir,
            mask_dir=cfg.dataset.mask_dir,
            batch_size=cfg.batch_size,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            augment=cfg.dataset.augment,
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
