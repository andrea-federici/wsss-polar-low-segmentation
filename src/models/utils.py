import torch
from torchinfo import summary
from src import models


def model_getter(model_name, cfg, print_summary=True):

    shared_hparams = {
        "num_labels": cfg.dataset.num_labels,
        "channels": len(cfg.dataset.channels),
    }

    if model_name == "unet":
        model = models.unet.Unet(**shared_hparams, **cfg.model.hparams)
    elif model_name == "unetformer":
        model = models.unetformer.FTUNetFormer(**shared_hparams)
    elif model_name == "segformer":
        model = models.segformer.get_model(**shared_hparams, **cfg.model.hparams)
    elif model_name == "upernet":
        model = models.upernet.get_model(
            **shared_hparams, **cfg.model.hparams, image_size=cfg.dataset.height
        )
    elif model_name == "dpt":
        model = models.dpt.get_model(
            **shared_hparams, **cfg.model.hparams, image_size=cfg.dataset.height
        )
    elif model_name == "smp":
        model = models.smp_models.get_model(
            **shared_hparams,
            **cfg.model.hparams,
        )
    else:
        raise NotImplementedError(f"[model getter]: Invalid model {model_name}")

    if print_summary:
        summary(model)
    return model


def handle_outputs(outputs, x, model_nametag: str):
    if model_nametag == "___segformer":
        outputs = torch.nn.functional.interpolate(
            outputs["logits"], size=x.shape[-2:], mode="bilinear", align_corners=False
        )
    if model_nametag in ["___dpt", "___uper"]:
        outputs = outputs["logits"]
    return outputs
