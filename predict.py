import numpy as np
import torch
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
import glob
import argparse
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
sys.path.append('../')
from neptune.utils import stringify_unsupported
from sklearn.metrics import jaccard_score, accuracy_score
from source import models, data, utils 

utils.misc.register_resolvers()
utils.misc.reduce_precision()


# TODO: this script was not updated and will not work as is.

@hydra.main(version_base=None, config_path="config", config_name="predict")
def run(cfg : DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Ensure that the checkpoint path was specified
    if not cfg.checkpoint.path:
        raise ValueError("No checkpoint was specified. Please specify the "
        "checkpoint path in the config file or from the command line "
        "when running the script.")
    
    checkpoint_path = os.path.join(
        cfg.checkpoint.base_folder,
        cfg.checkpoint.path
    )

    # Ensure that the checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    print(f"Loaded checkpoint: {checkpoint_path}")
    
    # Model
    model = models.utils.model_getter(cfg.model.name, cfg, print_summary=False)
    
    # Loss
    loss_fn = utils.seg_losses.loss_getter(name=cfg.loss.name, **cfg.loss.hparams)

    # Optim scheduler
    if cfg.get('lr_scheduler') is not None:
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
        plot_dict=dict(cfg.plot_preds_at_epoch)
    )
    
    model = seg.model
    model.eval()
            
    # Data module:
    data_module = data.data_loaders.VOCDataModule(
        data_dir="data",
        batch_size=cfg.batch_size,
        height=cfg.dataset.height,
        width=cfg.dataset.width,
        num_workers=cfg.workers
    )
    data_module.prepare_data()
    data_module.setup(stage='test')
    dl = data_module.test_dataloader()

    # cache_dir = os.path.join(data.data_loaders.CACHE_DIR)
    # data_module = data.data_loaders.AvalancheDataModule(
    #     cache_dir=cache_dir, 
    #     channels=cfg.dataset.channels, 
    #     batch_size=cfg.batch_size, 
    #     width=cfg.dataset.width, 
    #     height=cfg.dataset.height, 
    #     recache=False,
    #     augment=cfg.dataset.augment,
    #     num_workers=cfg.workers
    # )
    # dl = data_module.predict_dataloader()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dl):
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            # y_pred is of type 
            # transformers.modeling_outputs.SemanticSegmenterOutput and 
            # y_pred.logits is a torch.Tensor of shape [32, 21, 64, 64]. 
            # y is a torch.Tensor of shape [32, 256, 256].
            logits = y_pred.logits

            y = torch.nn.functional.interpolate(
                y.unsqueeze(1).float(), # interpolate() expects a 4D tensor
                size=logits.shape[-2:], # resize to same H,W as logits
                mode='nearest'
            ).squeeze(1) # remove the added channel dimension
            loss = loss_fn(logits, y.long())

            preds = torch.argmax(logits, dim=1).int()
            target = y.int()

            # Ignore void pixels for metrics only
            mask = target != 255
            preds_for_metrics = preds[mask].cpu().numpy()
            target_for_metrics = target[mask].cpu().numpy()
            
            mean_iou = jaccard_score(
                target_for_metrics.flatten(), 
                preds_for_metrics.flatten(),
                average='macro'
            )

            cmap = plt.get_cmap("tab20", np.max(target)+1) # 21 classes
            print(f"Number of classes used for cmap: {np.max(target)+1}")

            for idx in range(preds.shape[0]):
                plt.figure(figsize=(12, 4))

                # Plot predicted mask
                plt.subplot(1, 2, 1)
                plt.imshow(preds[idx].cpu().numpy(), cmap=cmap)
                plt.title("Predicted Mask")
                plt.axis("off")

                # Plot ground truth mask
                plt.subplot(1, 2, 2)
                plt.imshow(target[idx].cpu().numpy(), cmap=cmap)
                plt.title("Ground Truth Mask")
                plt.axis("off")

                plt.savefig(
                    os.path.join(
                        output_dir, 
                        f"segmentation_comparison_{batch_idx}_{idx}.png"
                    ),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()

            print(f"Batch {batch_idx}: Loss: {loss.item()}, Mean IoU: "
                  f"{mean_iou}")

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