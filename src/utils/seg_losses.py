from typing import Optional, Sequence, Union

import torch
from monai.losses.dice import DiceCELoss


def loss_getter(
    name: str,
    class_weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    dice_w: float = 0.5,
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
    else:
        raise ValueError(f"Unsupported loss function: {name}")
