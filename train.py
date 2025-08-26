import os

import torch
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import sys
from omegaconf import DictConfig, OmegaConf
import hydra

sys.path.append("../")
import neptune
from neptune.utils import stringify_unsupported
from dotenv import load_dotenv

from src import models, data, utils
import src.lightning_modules as lm
from src.utils.neptune_utils import _FilterCallback

neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)

utils.misc.register_resolvers()
utils.misc.reduce_precision()


@hydra.main(version_base=None, config_path="config", config_name="default")
def run(cfg: DictConfig) -> float:

    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Ensure that the dataset specified is supported
    supported_datasets = [
        utils.constants.PL_DATASET_NAME,
    ]
    if cfg.dataset.name not in supported_datasets:
        raise ValueError(
            f"Dataset {cfg.dataset.name} is not supported. "
            f"Supported datasets are: {', '.join(supported_datasets)}"
        )

    # Data module:
    if cfg.dataset.name == utils.constants.PL_DATASET_NAME:
        data_module = data.data_loader.DataModuleWrapper(
            image_dir=cfg.dataset.image_dir,
            mask_dir=cfg.dataset.mask_dir,
            batch_size=cfg.batch_size,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            augment=cfg.dataset.augment,
            num_workers=cfg.workers,
        )

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

    # Lightning module
    if cfg.dataset.name in [
        utils.constants.PL_DATASET_NAME,
        utils.constants.CANCER_DATASET_NAME,
    ]:
        seg = lm.custom_module.Segmentation(
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

    # Logger:
    if cfg.logger.backend == "tensorboard":
        logger = TensorBoardLogger(cfg.logger.logdir)
    elif cfg.logger.backend == "neptune":
        # Check if API token should be loaded from .env
        if getattr(cfg.logger, "use_env_token", False):
            load_dotenv()
            neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")
        else:
            neptune_api_token = None

        logger = utils.neptune_utils.NeptuneLogger(
            api_key=neptune_api_token,
            project_name=cfg.logger.project,
            save_dir=cfg.logger.logdir,
            tags=cfg.tags,
            params=stringify_unsupported(OmegaConf.to_container(cfg, resolve=True)),
            debug=cfg.logger.offline,
        )
        OmegaConf.save(cfg, "run_config.yaml")
        logger.log_artifact("run_config.yaml", delete_after=True)
    else:
        raise NotImplementedError("Backend not in ['tensorboard','neptune']")

    monitor = cfg.monitor
    mode = cfg.mode

    # Callbacks:
    early_stop_callback = EarlyStopping(
        monitor=monitor, patience=cfg["patience"], mode=mode
    )
    cb = [early_stop_callback]

    if cfg.checkpoints:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath=cfg.logger.logdir + "/checkpoints/",
            filename=model.nametag + "___{epoch:03d}-{val_f1:e}",
        )
        cb.append(checkpoint_callback)

    # Training:
    trainer = lightning.Trainer(
        logger=logger,
        callbacks=cb,
        devices=utils.misc.find_devices(1),
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        gradient_clip_val=cfg.clip_val,
        accelerator="gpu",
        overfit_batches=0.0,  # >0 for debug
    )
    trainer.fit(seg, datamodule=data_module)

    logger.finalize("success")

    return trainer.callback_metrics[monitor].item()


if __name__ == "__main__":
    run()
