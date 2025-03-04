import torch.nn.functional as F
import torch
from typing import Optional
from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss, TverskyLoss


def loss_getter(name: str = None,
                pos_weight: Optional[float] = 1.0,
                dice_w: Optional[float] = 0.5,
                squared_pred: Optional[bool] = False,
                **kwargs):
        
    if name=='custom_dice_ce':
        return lambda x, y: dice_ce_loss(x, y, class_weights=torch.tensor([1.0, pos_weight]).to(x.device), 
                                         squared_pred=squared_pred, dice_w=dice_w, **kwargs)
    if name=='dice_ce':
        loss = DiceCELoss_wrap(softmax=True, reduction='mean', squared_pred=squared_pred, lambda_dice=dice_w, 
                          lambda_ce=1.0-dice_w, to_onehot_y=True, **kwargs)
        return lambda x, y: loss(x, y.unsqueeze(1), ce_weight=torch.tensor([1.0, pos_weight]).to(x.device))
    elif name=='dice_focal':
        loss = DiceFocalLoss(softmax=True, reduction='mean', to_onehot_y=True, squared_pred=squared_pred,
                             focal_weight=[1.0, pos_weight],  lambda_dice=dice_w, 
                             lambda_focal=1.0-dice_w,**kwargs)
        return lambda x, y: loss(x, y.unsqueeze(1))
    elif name=='general_dice_focal':
        loss = GeneralizedDiceFocalLoss(softmax=True, reduction='mean', to_onehot_y=True,
                                        focal_weight=[1.0, pos_weight], lambda_gdl=dice_w, 
                                        lambda_focal=1.0-dice_w, **kwargs)
        return lambda x, y: loss(x, y.unsqueeze(1))
    elif name=='tversky':
        # FNs should be weighted more than FPs in highly imbalanced dataset
        # alpha: weight of FP -- beta: weight of FN
        loss = TverskyLoss(softmax=True, reduction='mean', to_onehot_y=True, alpha=0.3, beta=0.7)
        return lambda x, y: loss(x, y.unsqueeze(1))
    
    elif name=='dice_ce_multiclass':
        return lambda x, y: dice_ce_loss_multiclass(x, y, class_weights=None, 
                                                    ignore_index=255, dice_w=dice_w)
    else:
        raise NotImplementedError(f"Loss {name} not implemented")


def dice_ce_loss(predictions, 
                ground_truths, 
                class_weights=None, 
                dice_w=0.5, 
                num_classes=2, 
                dims=(1, 2), 
                smooth=1e-5,
                squared_pred=False):
    """
    Smooth Dice coefficient + Cross-entropy loss 
    """
    
    CE_loss = F.cross_entropy(predictions, ground_truths, 
                              weight=class_weights.to(predictions.device))
    
    if dice_w > 0:
        ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
        prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

        intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
        if squared_pred:
            summation = torch.sum(prediction_norm**2, dim=dims) + torch.sum(ground_truth_oh**2, dim=dims)
        else:
            summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)

        dice = (2.0 * intersection + smooth) / (summation + smooth)
        # dice = torch.matmul(dice, class_weights.to(predictions.device))
        dice_loss = 1.0 - dice.mean()        
        return dice_w*dice_loss + (1.0 - dice_w)*CE_loss
    
    else:
        return CE_loss
    

class DiceCELoss_wrap(DiceCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor, ce_weight: torch.Tensor):
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
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
    

def dice_ce_loss_multiclass(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    class_weights: torch.Tensor = None,
    dice_w: float = 0.5,
    ignore_index: int = 255,
    smooth: float = 1e-5,
    squared_pred: bool = False
):
    """
    A multi-class version of dice + cross-entropy loss.

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
        weight=class_weights,      # e.g. shape = [C], if you have class weighting
        ignore_index=ignore_index  # skip “void” label if your dataset uses 255
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
        ground_truths = torch.where(valid_mask, ground_truths, torch.zeros_like(ground_truths))

    B, C, H, W = predictions.shape

    # One-hot encode the ground-truth to [B, C, H, W]
    # e.g. if ground_truths in [0..C-1], then:
    gt_onehot = F.one_hot(ground_truths, num_classes=C)  # [B, H, W, C]
    gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()    # -> [B, C, H, W]

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
        gt_sum   = torch.sum(gt_onehot**2,  dim=(2, 3))
    else:
        pred_sum = torch.sum(pred_probs, dim=(2, 3))
        gt_sum   = torch.sum(gt_onehot, dim=(2, 3))

    dice_per_class = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    dice_loss = 1.0 - dice_per_class.mean()  # average over batch & classes

    # Combine
    return dice_w * dice_loss + (1.0 - dice_w) * CE_loss