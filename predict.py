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
from src import models, data, utils

utils.misc.register_resolvers()
utils.misc.reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="pl_config")
def run(cfg: DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    stage = "train"
    do_augment = False
    save_only_pos = True
    binarize_masks = False
    top_classes = -1  # Use -1 to skip this step

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
    loss = utils.seg_losses.loss_getter(name=cfg.loss.name, **cfg.loss.hparams)

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
        loss=loss,
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
    dataloader = _get_dataloader(cfg, stage, do_augment=do_augment)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    all_pred = []
    all_target = []
    total_loss = 0.0
    batch_count = 0

    tp = tn = fp = fn = 0

    with torch.no_grad():
        for batch_idx, (x, y, x_names) in enumerate(
            tqdm(dataloader, desc="Doing inference...")
        ):
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            outputs = models.utils.handle_outputs(outputs, x, model.nametag)

            loss = loss(outputs, y.long())
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

            # Ignore void pixels for metrics only
            mask = target != 255

            # Accumulate metrics
            all_pred.append(preds[mask].cpu().numpy().flatten())
            all_target.append(target[mask].cpu().numpy().flatten())
            total_loss += loss.item()
            batch_count += 1

            cmap = plt.get_cmap("viridis", cfg.dataset.num_labels)

            for i in range(preds.shape[0]):
                # Convert image from tensor (CHW) to numpy (HWC) and normalize
                img = x[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())

                # Get predictions
                pr_mask = preds[i].cpu().numpy()  # original prediction without CRF
                gt_mask = target[i].cpu().numpy()  # ground truth mask

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

                # Apply colormap (cmap expects values normalized by the number of labels)
                pred_colored_original = cmap(pr_mask / cfg.dataset.num_labels)[:, :, :3]
                target_colored = cmap(gt_mask / cfg.dataset.num_labels)[:, :, :3]

                # Override the background (label 0) to use the original image color
                background_mask_pred = pr_mask == 0
                pred_colored_original[background_mask_pred] = img[background_mask_pred]

                background_mask_target = gt_mask == 0
                target_colored[background_mask_target] = img[background_mask_target]

                # Create overlays by blending the original image with the colored masks
                overlay_pred = np.clip(
                    0.6 * img + 0.4 * pred_colored_original, 0.0, 1.0
                )
                overlay_target = np.clip(0.6 * img + 0.4 * target_colored, 0.0, 1.0)

                img_name = x_names[i].removesuffix(".png")

                ### --- PLOT COMPARISON --- ###
                plt.figure(figsize=(14, 4))

                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title(f"Original Image. GT: {gt_label}")
                plt.axis("off")

                # Prediction
                plt.subplot(1, 3, 2)
                plt.imshow(overlay_pred)
                plt.title("Prediction")
                plt.axis("off")

                # Pseudo label
                plt.subplot(1, 3, 3)
                plt.imshow(overlay_target)
                plt.title("Pseudo Label")
                plt.axis("off")

                ### --- SAVE IMAGES --- ###
                # Save figure
                plt.savefig(
                    os.path.join(output_dir, f"{img_name}_comparison.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                additional_dir = os.path.join(output_dir, "additional")
                os.makedirs(additional_dir, exist_ok=True)

                plt.imsave(
                    os.path.join(additional_dir, f"{img_name}_original.png"), img
                )
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
        mean_iou = jaccard_score(all_target, all_pred, average="macro")
        avg_loss = total_loss / batch_count if batch_count else 0.0

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Final Results - Avg Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}")


# TODO: do we want to keep this option to override the do_augment flag? Or do we just
# want to use the one from the config file?
def _get_dataloader(cfg: DictConfig, stage: str, do_augment: bool = None):
    supported_stages = ["train", "val", "test"]
    assert (
        stage in supported_stages
    ), f"Stage {stage} not supported. Supported stages: {', '.join(supported_stages)}"
    if cfg.dataset.name in [
        utils.constants.PL_DATASET_NAME,
        utils.constants.CANCER_DATASET_NAME,
    ]:
        data_module = data.data_loader.DataModuleWrapper(
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
