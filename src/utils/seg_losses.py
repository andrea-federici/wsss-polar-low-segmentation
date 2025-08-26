import warnings

import torch
from typing import Optional, Union, Sequence
from monai.losses.dice import (
    DiceCELoss,
    DiceFocalLoss,
    GeneralizedDiceFocalLoss,
)
from monai.losses.tversky import TverskyLoss


def loss_getter(
    name: str,
    class_weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    dice_w: float = 0.5,
    **kwargs,
) -> torch.nn.Module:
    if class_weight is not None and not isinstance(class_weight, torch.Tensor):
        class_weight = torch.tensor(class_weight, dtype=torch.float32)

    if name == "dice_ce":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            weight=class_weight,
            lambda_dice=dice_w,
            lambda_ce=1.0 - dice_w,
        )
    elif name == "dice_focal":
        return DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            weight=class_weight,
            lambda_dice=dice_w,
            lambda_focal=1.0 - dice_w,
        )
    elif name == "generalized_dice_focal":
        return GeneralizedDiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            weight=class_weight,
            lambda_gdl=dice_w,
            lambda_focal=1.0 - dice_w,
        )
    elif name == "tversky":
        if class_weight is not None:
            warnings.warn("class_weight is not used in TverskyLoss and will be ignored.")

        alpha = kwargs.get("alpha", 0.5)
        beta = kwargs.get("beta", 0.5)
        return TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"Unsupported loss function: {name}")
