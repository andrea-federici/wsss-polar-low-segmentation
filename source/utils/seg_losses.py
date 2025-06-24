import torch.nn.functional as F
import torch
from typing import Optional
from monai.losses import (
    DiceCELoss,
    DiceFocalLoss,
    GeneralizedDiceFocalLoss,
    TverskyLoss,
)


def loss_getter(
    name: str = None,
    class_weight: Optional[float] = None,
    dice_w: Optional[float] = 0.5,
    mse_w: Optional[float] = 0.1,
    squared_pred: Optional[bool] = False,
    **kwargs,
):

    if name == "custom_dice_ce":
        return lambda x, y: dice_ce_loss(
            predictions=x,
            ground_truths=y,
            class_weights=torch.tensor(class_weight).to(x.device),
            ignore_index=255,
            dice_w=dice_w,
            **kwargs,
        )

    elif name == "dice_ce_mse":
        return lambda x, y: dice_ce_mse_loss(
            predictions=x,
            ground_truths=y,
            class_weights=torch.tensor(class_weight).to(x.device),
            ignore_index=255,
            dice_w=dice_w,
            mse_w=mse_w,
            squared_pred=squared_pred,
            **kwargs,
        )

    elif name == "dice_ce":
        base_loss = DiceCELoss_wrap(
            softmax=True,
            reduction="mean",
            squared_pred=squared_pred,
            lambda_dice=dice_w,
            lambda_ce=1.0 - dice_w,
            to_onehot_y=True,
            **kwargs,
        )

        def wrapped_loss(x, y):
            # Some losses do not have an ignore_index parameter, we must handle 255 ourselves
            # x shape [B,C,H,W], y shape [B,H,W] with 255 for void
            # Replace void (255) with 0 (background)
            y = torch.where(y == 255, torch.zeros_like(y), y)
            return base_loss(
                x, y.unsqueeze(1), ce_weight=torch.tensor(class_weight).to(x.device)
            )

        return wrapped_loss

    elif name == "dice_focal":
        base_loss = DiceFocalLoss(
            softmax=True,
            reduction="mean",
            to_onehot_y=True,
            include_background=True,  # Assumes 0 is background
            squared_pred=squared_pred,
            weight=class_weight,
            lambda_dice=dice_w,
            lambda_focal=1.0 - dice_w,
            **kwargs,
        )

        def wrapped_loss(x, y):
            # Some losses do not have an ignore_index parameter, we must handle 255 ourselves
            # x shape [B,C,H,W], y shape [B,H,W] with 255 for void
            # Replace void (255) with 0 (background)
            y = torch.where(y == 255, torch.zeros_like(y), y)
            return base_loss(x, y.unsqueeze(1))

        return wrapped_loss

    elif name == "general_dice_focal":
        base_loss = GeneralizedDiceFocalLoss(
            softmax=True,
            reduction="mean",
            to_onehot_y=True,
            include_background=True,  # Assumes 0 is background
            weight=class_weight,
            lambda_gdl=dice_w,
            lambda_focal=1.0 - dice_w,
            **kwargs,
        )

        def wrapped_loss(x, y):
            # Some losses do not have an ignore_index parameter, we must handle 255 ourselves
            # x shape [B,C,H,W], y shape [B,H,W] with 255 for void
            # Replace void (255) with 0 (background)
            y = torch.where(y == 255, torch.zeros_like(y), y)
            return base_loss(x, y.unsqueeze(1))

        return wrapped_loss

    elif name == "tversky":
        # FNs should be weighted more than FPs in highly imbalanced dataset
        # alpha: weight of FP -- beta: weight of FN
        base_loss = TverskyLoss(
            softmax=True,
            reduction="mean",
            to_onehot_y=True,
            include_background=True,  # Assumes 0 is background
            **kwargs,
        )

        def wrapped_loss(x, y):
            # Some losses do not have an ignore_index parameter, we must handle 255 ourselves
            # x shape [B,C,H,W], y shape [B,H,W] with 255 for void
            # Replace void (255) with 0 (background)
            y = torch.where(y == 255, torch.zeros_like(y), y)
            return base_loss(x, y.unsqueeze(1))

        return wrapped_loss

    else:
        raise NotImplementedError(f"Loss {name} not implemented")


class DiceCELoss_wrap(DiceCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, ce_weight: torch.Tensor
    ):
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)

        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        ce_loss = F.cross_entropy(input, target, weight=ce_weight.to(target.device))
        total_loss: torch.Tensor = (
            self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        )

        return total_loss


def dice_ce_loss(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    class_weights: torch.Tensor = None,
    dice_w: float = 0.5,
    ignore_index: int = 255,
    smooth: float = 1e-5,
    squared_pred: bool = False,
):
    """
    A custom version of dice + cross-entropy loss supporting multi-classes.

    Args:
        predictions: shape [B, C, H, W]
        ground_truths: shape [B, H, W], values in [0..C-1] or possibly 255 for 'void' pixels
        class_weights: length=C weight vector for cross_entropy, or None
        dice_w: how much dice contributes to final loss (0.0 -> pure CE, 1.0 -> pure Dice)
        ignore_index: label to ignore in CrossEntropy (255 in VOC)
        smooth: smoothing constant for dice
        squared_pred: if True, use squared probabilities in the denominator
                      (slightly changes dice gradient)
    Returns:
        A scalar tensor representing the combined dice+CE loss.
    """

    # (1) Cross-entropy loss for multi-class.
    #     ignore_index=255 means the 'void' labeled pixels in VOC won't affect CE
    CE_loss = F.cross_entropy(
        predictions,
        ground_truths,
        weight=class_weights,  # e.g. shape = [C], if you have class weighting
        ignore_index=ignore_index,  # skip “void” label if your dataset uses 255
    )

    # If we do not want a dice component, just return CE.
    if dice_w <= 0:
        return CE_loss

    # (2) Dice loss for multi-class
    #     We must ignore 'void' pixels in the dice portion too if we want consistent metrics.
    #     The simplest approach is to create a mask for valid pixels:
    valid_mask = None
    if ignore_index is not None:
        valid_mask = ground_truths != ignore_index
        # Replace 'void' with 0 just so one_hot won't exceed index range
        ground_truths = torch.where(
            valid_mask, ground_truths, torch.zeros_like(ground_truths)
        )

    B, C, H, W = predictions.shape

    # One-hot encode the ground-truth to [B, C, H, W]
    # e.g. if ground_truths in [0..C-1], then:
    gt_onehot = F.one_hot(ground_truths, num_classes=C)  # [B, H, W, C]
    gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()  # -> [B, C, H, W]

    # Probability map via softmax across channels
    pred_probs = F.softmax(predictions, dim=1)  # shape [B, C, H, W]

    # If ignoring void, zero out predictions where mask is invalid
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(1)  # shape [B, 1, H, W]
        pred_probs = pred_probs * valid_mask
        gt_onehot = gt_onehot * valid_mask

    # Intersection = sum(pred * gt), Denominator = sum(pred) + sum(gt)
    intersection = torch.sum(pred_probs * gt_onehot, dim=(2, 3))  # [B, C]
    if squared_pred:
        pred_sum = torch.sum(pred_probs**2, dim=(2, 3))
        gt_sum = torch.sum(gt_onehot**2, dim=(2, 3))
    else:
        pred_sum = torch.sum(pred_probs, dim=(2, 3))
        gt_sum = torch.sum(gt_onehot, dim=(2, 3))

    dice_per_class = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    dice_loss = 1.0 - dice_per_class.mean()  # average over batch & classes

    # Combine
    return dice_w * dice_loss + (1.0 - dice_w) * CE_loss


# TODO: needs cleanup
def dice_ce_mse_loss(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    class_weights: torch.Tensor = None,
    dice_w: float = 0.5,
    mse_w: float = 0.1,
    ignore_index: int = 255,
    smooth: float = 1e-5,
    squared_pred: bool = False,
):
    """
    Combines the Dice+CE loss with an additional MSE loss that captures
    the ordinal relationship among labels.
    """
    # First, compute the standard Dice+CE loss
    loss_dice_ce = dice_ce_loss(
        predictions,
        ground_truths,
        class_weights=class_weights,
        dice_w=dice_w,
        ignore_index=ignore_index,
        smooth=smooth,
        squared_pred=squared_pred,
    )

    # Compute the MSE component.
    # Here we convert the network's output into an expectation over class labels.
    # Assuming the classes are ordered (0, 1, 2, ...), we compute the expected label.
    pred_probs = F.softmax(predictions, dim=1)  # [B, C, H, W]
    num_classes = predictions.shape[1]

    # Create an index tensor with values 0, 1, ..., num_classes-1
    indices = torch.arange(
        num_classes, device=predictions.device, dtype=predictions.dtype
    )

    # Compute the expectation for each pixel: sum_c (c * p(c))
    pred_expectation = (pred_probs * indices.view(1, -1, 1, 1)).sum(dim=1)  # [B, H, W]

    # Prepare the mask to ignore void labels
    valid_mask = ground_truths != ignore_index

    # For MSE, only consider valid pixels
    if valid_mask.sum() > 0:
        mse_loss = F.mse_loss(
            pred_expectation[valid_mask],
            ground_truths[valid_mask].to(pred_expectation.dtype),
        )
    else:
        mse_loss = torch.tensor(0.0, device=predictions.device)

    total_loss = (1 - mse_w) * loss_dice_ce + mse_w * mse_loss
    return total_loss
