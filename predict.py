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
sys.path.append('../')
from neptune.utils import stringify_unsupported
from source import models, data, utils 

utils.misc.register_resolvers()
utils.misc.reduce_precision()


# TODO: ths script was not updated and will not work as is.

def run(checkpoint_path): 
    @hydra.main(version_base=None, config_path="config", config_name="predict")
    def _run(cfg : DictConfig) -> float:
        
        print(OmegaConf.to_yaml(cfg, resolve=True))
        
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
        cache_dir = os.path.join(data.data_loaders.CACHE_DIR)
        data_module = data.data_loaders.AvalancheDataModule(
            cache_dir=cache_dir, 
            channels=cfg.dataset.channels, 
            batch_size=cfg.batch_size, 
            width=cfg.dataset.width, 
            height=cfg.dataset.height, 
            recache=False,
            augment=cfg.dataset.augment,
            num_workers=cfg.workers
        )
        dl = data_module.predict_dataloader()
        
        with torch.no_grad():
            for x,y in dl: 
                y_hat = model(x.to('cuda'))
                
                break
        
        return 
        
    return _run
    
# if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config-path')
#     args, unknown = parser.parse_known_args()
#     checkpoint_path = glob.glob(os.path.abspath(os.path.join(args.config_path, '..', 'checkpoints', '*.ckpt')))[0]
#     print(checkpoint_path)
#     assert os.path.exists(checkpoint_path), f'checkpoint does not exist: {checkpoint_path}'
#     run(checkpoint_path)()