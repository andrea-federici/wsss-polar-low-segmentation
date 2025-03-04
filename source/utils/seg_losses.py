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