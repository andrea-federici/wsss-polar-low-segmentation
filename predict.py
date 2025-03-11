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
from source import models, data, utils 

utils.misc.register_resolvers()
utils.misc.reduce_precision()


# TODO: ths script was not updated and will not work as is.

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
    
    with torch.no_grad():
        for x,y in dl:
            y_hat = model(x.to('cuda'))
            # y_hat is of type 
            # transformers.modeling_outputs.SemanticSegmenterOutput and 
            # y_hat.logits is a torch.Tensor of shape [32, 21, 64, 64]. 
            # y is a torch.Tensor of shape [32, 256, 256].
            logits = y_hat.logits
            predicted_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            ground_truth_mask = y.squeeze(0).cpu().numpy()

            for idx in range(32):
                plt.figure(figsize=(12, 4))

                # Plot predicted mask
                plt.subplot(1, 2, 1)
                plt.imshow(predicted_mask[idx], cmap="jet")
                plt.title("Predicted Mask")
                plt.axis("off")

                # Plot ground truth mask
                plt.subplot(1, 2, 2)
                plt.imshow(ground_truth_mask[idx], cmap="jet")
                plt.title("Ground Truth Mask")
                plt.axis("off")

                plt.savefig(f"segmentation_comparison{idx}.png", dpi=300, bbox_inches="tight")

            plt.close()

            print(f"y shape: {y.shape}")
            print(f"y hat logits shape: {y_hat.logits.shape}")
            print(f"y_hat type: {type(y_hat)}")
            
            break # Remember to remove this

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