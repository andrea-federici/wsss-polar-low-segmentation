import torch
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
import cv2

sys.path.append("../")
from sklearn.metrics import jaccard_score
from source import models, data, utils

utils.misc.register_resolvers()
utils.misc.reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="predict")
def run(cfg: DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Ensure that the dataset specified is supported
    if cfg.dataset.name not in [
        utils.constants.VOC_DATASET_NAME,
        utils.constants.PL_DATASET_NAME,
    ]:
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. "
            f"Supported datasets: {utils.constants.VOC_DATASET_NAME}, "
            f"{utils.constants.PL_DATASET_NAME}"
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

    print(f"Loaded checkpoint: {checkpoint_path}")

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

    seg = models.sar_module.Segmentation.load_from_checkpoint(
        checkpoint_path,
        model=model,
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
    model.eval()

    # Data module:
    if cfg.dataset.name == utils.constants.VOC_DATASET_NAME:
        data_module = data.data_loaders.VOCDataModule(
            data_dir="data",
            batch_size=cfg.batch_size,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            num_workers=cfg.workers,
        )
    elif cfg.dataset.name == utils.constants.PL_DATASET_NAME:
        data_module = data.pl_loader.PLDataModule(
            image_dir=cfg.dataset.image_dir,
            mask_dir=cfg.dataset.mask_dir,
            batch_size=cfg.batch_size,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            augment=cfg.dataset.augment,
            num_workers=cfg.workers,
        )
    data_module.prepare_data()
    data_module.setup(stage="test")
    dl = data_module.val_dataloader()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dl):
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            outputs = models.utils.handle_outputs(outputs, x, model.nametag)

            loss = loss_fn(outputs, y.long())
            preds = torch.argmax(outputs, dim=1).int()
            target = y.int()

            # Convert logits or soft outputs to probability maps
            # Assuming outputs is logits: apply softmax along the class dimension
            softmax_outputs = torch.softmax(outputs, dim=1)

            # Optional: Convert probabilities to CPU numpy arrays for CRF processing
            # Process each image in the batch
            refined_preds = []
            for idx in range(softmax_outputs.shape[0]):
                # Get the original image and predicted probability map
                orig_img = x[idx].cpu().permute(1, 2, 0).numpy()
                orig_img = (orig_img - orig_img.min()) / (
                    orig_img.max() - orig_img.min()
                )
                prob_map = (
                    softmax_outputs[idx].cpu().numpy()
                )  # shape: (num_classes, H, W)

                # Here, we assume binary segmentation for CRF.
                # For binary, prob_map[0] is background and prob_map[1] is foreground.
                refined_mask = utils.crf_utils.refine_mask_with_crf(
                    orig_img, prob_map, num_classes=cfg.dataset.num_labels, iterations=5
                )
                refined_preds.append(refined_mask)

            # Ignore void pixels for metrics only
            mask = target != 255
            preds_for_metrics = preds[mask].cpu().numpy()
            target_for_metrics = target[mask].cpu().numpy()

            mean_iou = jaccard_score(
                target_for_metrics.flatten(),
                preds_for_metrics.flatten(),
                average="macro",
            )

            # TODO: use color map from misc.img_logging
            cmap = plt.get_cmap("viridis", cfg.dataset.num_labels)  # VOC -> 21 classes

            for idx in range(preds.shape[0]):
                # Convert image from tensor (CHW) to numpy (HWC) and normalize
                original_img = x[idx].cpu().permute(1, 2, 0).numpy()
                original_img = (original_img - original_img.min()) / (
                    original_img.max() - original_img.min()
                )

                # Get predictions
                pred_mask_original = (
                    preds[idx].cpu().numpy()
                )  # original prediction without CRF
                pred_mask_crf = refined_preds[idx]  # CRF refined mask
                target_mask = target[idx].cpu().numpy()  # ground truth mask

                # Resize masks to match original image size
                pred_resized_original = cv2.resize(
                    pred_mask_original,
                    (original_img.shape[1], original_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred_resized_crf = cv2.resize(
                    pred_mask_crf,
                    (original_img.shape[1], original_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                target_resized = cv2.resize(
                    target_mask,
                    (original_img.shape[1], original_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Apply colormap (cmap expects values normalized by the number of labels)
                pred_colored_original = cmap(
                    pred_resized_original / cfg.dataset.num_labels
                )[:, :, :3]
                pred_colored_crf = cmap(pred_resized_crf / cfg.dataset.num_labels)[
                    :, :, :3
                ]
                target_colored = cmap(target_resized / cfg.dataset.num_labels)[:, :, :3]

                # Override the background (label 0) to use the original image color
                background_mask_pred = pred_resized_original == 0
                pred_colored_original[background_mask_pred] = original_img[
                    background_mask_pred
                ]

                background_mask_crf = pred_resized_crf == 0
                pred_colored_crf[background_mask_crf] = original_img[
                    background_mask_crf
                ]

                background_mask_target = target_resized == 0
                target_colored[background_mask_target] = original_img[
                    background_mask_target
                ]

                # Create overlays by blending the original image with the colored masks
                overlay_original = 0.6 * original_img + 0.4 * pred_colored_original
                overlay_crf = 0.6 * original_img + 0.4 * pred_colored_crf
                overlay_target = 0.6 * original_img + 0.4 * target_colored

                # Create a three-panel figure
                plt.figure(figsize=(18, 4))

                plt.subplot(1, 4, 1)
                plt.imshow(original_img)
                plt.title("Original Image")
                plt.axis("off")

                # Plot original prediction (without CRF)
                plt.subplot(1, 4, 2)
                plt.imshow(overlay_original)
                plt.title("Prediction (no post-processing)")
                plt.axis("off")

                # Plot CRF refined prediction
                plt.subplot(1, 4, 3)
                plt.imshow(overlay_crf)
                plt.title("CRF Refined Prediction")
                plt.axis("off")

                # Plot ground truth mask
                plt.subplot(1, 4, 4)
                plt.imshow(overlay_target)
                plt.title("Pseudo Label")
                plt.axis("off")

                # Save and close the figure
                plt.savefig(
                    os.path.join(
                        output_dir, f"segmentation_comparison_{batch_idx}_{idx}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            print(f"Batch {batch_idx}: Loss: {loss.item()}, Mean IoU: " f"{mean_iou}")

            # print(f"y shape: {y.shape}")
            # print(f"y_pred logits shape: {y_pred.logits.shape}")
            # print(f"y_pred type: {type(y_pred)}")


if __name__ == "__main__":
    run()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config-path')
#     args, unknown = parser.parse_known_args()
#     checkpoint_path = glob.glob(os.path.abspath(os.path.join(args.config_path, '..', 'checkpoints', '*.ckpt')))[0]
#     print(checkpoint_path)
#     assert os.path.exists(checkpoint_path), f'checkpoint does not exist: {checkpoint_path}'
#     run(checkpoint_path)()
