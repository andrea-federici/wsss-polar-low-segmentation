import warnings
from typing import Optional, Sequence, Union

import torch
from monai.losses.dice import DiceCELoss, DiceFocalLoss, DiceLoss, GeneralizedDiceFocalLoss
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
            softmax=True,  # TODO: check that this works also for binary masks. Or in that case do we need to set sigmoid=True?
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
    elif name == "soft_dice_bce":
        if class_weight is not None:
            warnings.warn("class_weight is not used in SoftDiceBCELoss and will be ignored.")

        return SoftDiceBCELoss(
            dice_w=dice_w,
            pos_weight=None,
            to_onehot_y=False,
            sigmoid=True,
            soft_label=True,
        )
    else:
        raise ValueError(f"Unsupported loss function: {name}")


class SoftDiceBCELoss(torch.nn.Module):
    def __init__(
        self,
        dice_w: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,  # Used by BCEWithLogitsLoss
        to_onehot_y: bool = False,
        sigmoid: bool = True,  # Set this to True if we are passing logits
        soft_label: bool = True,
    ):
        super().__init__()

        if pos_weight is not None:
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = torch.nn.BCEWithLogitsLoss()

        self.dice = DiceLoss(
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            soft_label=soft_label,
        )
        self.dice_w = dice_w

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, soft_targets)
        dice_loss = self.dice(logits, soft_targets)
        return (1.0 - self.dice_w) * bce_loss + self.dice_w * dice_loss
