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
    ignore_index: int = None,
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

    # --- Cross-Entropy Loss ---
    ce_loss = F.cross_entropy(
        predictions,
        ground_truths,
        weight=class_weights,  # e.g. shape = [C], if you have class weighting
        ignore_index=ignore_index,  # skip “void” label if your dataset uses 255
    )

    # If no Dice contribution, return CE directly
    if dice_w <= 0:
        return ce_loss

    # --- Prepare ground truth for Dice ---
    if ignore_index is not None:
        valid_mask = ground_truths != ignore_index
        ground_truths = torch.where(
            valid_mask, ground_truths, torch.zeros_like(ground_truths)
        )
    else:
        valid_mask = torch.ones_like(ground_truths, dtype=torch.bool)

    _, C, _, _ = predictions.shape

    # One-hot encode the ground-truth to [B, C, H, W]
    gt_onehot = F.one_hot(ground_truths, num_classes=C)  # [B, H, W, C]
    gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()  # -> [B, C, H, W]

    # Probability map via softmax across channels
    pred_probs = F.softmax(predictions, dim=1)  # shape [B, C, H, W]

    # Zero out predictions where mask is invalid
    valid_mask = valid_mask.unsqueeze(1)  # shape [B, 1, H, W]
    pred_probs = pred_probs * valid_mask
    gt_onehot = gt_onehot * valid_mask

    # --- Dice computation ---
    intersection = torch.sum(pred_probs * gt_onehot, dim=(2, 3))  # [B, C]
    if squared_pred:
        pred_sum = torch.sum(pred_probs**2, dim=(2, 3))
        gt_sum = torch.sum(gt_onehot**2, dim=(2, 3))
    else:
        pred_sum = torch.sum(pred_probs, dim=(2, 3))
        gt_sum = torch.sum(gt_onehot, dim=(2, 3))

    dice_per_class = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)

    # --- Apply class weights if provided ---
    if class_weights is not None:
        class_weights = class_weights.to(predictions.device).float()
        class_weights = class_weights / class_weights.sum()  # normalize
        dice_loss = (
            1.0 - (dice_per_class * class_weights).sum(dim=1).mean()
        )  # weighted avg over classes, then mean over batch
    else:
        dice_loss = 1.0 - dice_per_class.mean()

    # Weighted combination
    return dice_w * dice_loss + (1.0 - dice_w) * ce_loss


def dice_ce_mse_loss(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    dice_w: float = 0.5,
    mse_w: float = 0.5,
    ce_w: float = 0.5,
    ignore_index: Optional[int] = 255,
    smooth: float = 1e-5,
    squared_pred: bool = False,
) -> torch.Tensor:
    """
    Compute loss = alpha * dice + beta * ce + gamma * mse,
    where alpha, beta, gamma are normalized weights derived from dice_w, ce_w, mse_w.

    - If ce_w is None: ce_w = 1 - dice_w (backwards-compatible default).
    - class_weights (if provided) is applied to CE and to Dice (Dice uses normalized class weights).
    """

    # --- prepare devices / dtypes / shapes ---
    device = predictions.device
    dtype = predictions.dtype
    B, C, H, W = predictions.shape

    # Normalize the three weights so they sum to 1 (so they're comparable)
    if dice_w < 0 or ce_w < 0 or mse_w < 0:
        raise ValueError("Weights (dice_w, ce_w, mse_w) must be > 0")
    w_sum = float(dice_w + ce_w + mse_w)
    alpha = float(dice_w) / w_sum
    beta = float(ce_w) / w_sum
    gamma = float(mse_w) / w_sum

    # Ensure ground truth is integer type for CE/one-hot
    if not torch.is_floating_point(ground_truths):
        # likely long already; keep as is
        gt_int = ground_truths
    else:
        gt_int = ground_truths.long()

    # --- Cross-Entropy term (beta * CE) ---
    # Move class_weights to device/dtype expected by cross_entropy
    ce_weight = class_weights.to(device).float() if class_weights is not None else None
    ce_loss = F.cross_entropy(
        predictions, gt_int, weight=ce_weight, ignore_index=ignore_index
    )

    # If alpha == 0 and gamma == 0 => only CE needed (fast path)
    if alpha == 0.0 and gamma == 0.0:
        return ce_loss

    # --- Prepare mask and safe GT for Dice and MSE ---
    if ignore_index is None:
        valid_mask = torch.ones_like(gt_int, dtype=torch.bool, device=device)
        gt_safe = gt_int
    else:
        valid_mask = gt_int != ignore_index
        # replace invalid labels with 0 to keep one_hot happy
        gt_safe = torch.where(valid_mask, gt_int, torch.zeros_like(gt_int))

    # --- Dice term (alpha * Dice) ---
    if alpha > 0.0:
        gt_onehot = (
            F.one_hot(gt_safe, num_classes=C).permute(0, 3, 1, 2).float()
        )  # [B,C,H,W]
        pred_probs = F.softmax(predictions, dim=1)

        vm = valid_mask.unsqueeze(1)  # [B,1,H,W]
        pred_probs_masked = pred_probs * vm
        gt_onehot_masked = gt_onehot * vm

        intersection = torch.sum(
            pred_probs_masked * gt_onehot_masked, dim=(2, 3)
        )  # [B,C]
        if squared_pred:
            pred_sum = torch.sum(pred_probs_masked**2, dim=(2, 3))
            gt_sum = torch.sum(gt_onehot_masked**2, dim=(2, 3))
        else:
            pred_sum = torch.sum(pred_probs_masked, dim=(2, 3))
            gt_sum = torch.sum(gt_onehot_masked, dim=(2, 3))

        dice_per_class = (2.0 * intersection + smooth) / (
            pred_sum + gt_sum + smooth
        )  # [B,C]

        if class_weights is not None:
            cw = class_weights.to(device).float()
            # avoid division by zero if sum is zero; add tiny epsilon (but class_weights should be >0)
            cw_sum = cw.sum().item()
            if cw_sum == 0:
                normalized_cw = cw
            else:
                normalized_cw = cw / cw_sum
            # weighted over classes, then mean over batch
            # dice_per_class: [B,C] ; normalized_cw: [C] -> multiply broadcast -> [B,C]
            dice_score_per_batch = (dice_per_class * normalized_cw.view(1, -1)).sum(
                dim=1
            )  # [B]
            dice_loss = 1.0 - dice_score_per_batch.mean()
        else:
            dice_loss = 1.0 - dice_per_class.mean()
    else:
        dice_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # --- MSE term (gamma * MSE) ---
    if gamma > 0.0:
        pred_probs = F.softmax(predictions, dim=1)  # [B,C,H,W]
        indices = torch.arange(C, device=device, dtype=dtype)
        pred_expectation = (pred_probs * indices.view(1, -1, 1, 1)).sum(
            dim=1
        )  # [B,H,W]

        # MSE computed only over valid pixels
        if valid_mask.any().item():
            gt_for_mse = gt_safe.to(
                dtype
            )  # safe because invalids were set to 0 but masked out
            mse_loss = F.mse_loss(pred_expectation[valid_mask], gt_for_mse[valid_mask])
        else:
            mse_loss = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        mse_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # --- Combine final losses with normalized weights ---
    total = alpha * dice_loss + beta * ce_loss + gamma * mse_loss
    return total
