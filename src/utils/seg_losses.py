from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


class SimpleDiceCELoss(torch.nn.Module):
    """Fallback implementation of Dice + Cross-Entropy loss."""

    def __init__(
        self,
        *,
        weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 0.5,
        lambda_ce: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        if weight is not None:
            if weight.dim() != 1:
                raise ValueError("weight tensor must be 1-dimensional")
            self.register_buffer("_weight", weight.float())
        else:
            self._weight = None
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim() and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dtype != torch.long:
            target = target.long()

        num_classes = logits.shape[1]
        log_probs = torch.log_softmax(logits, dim=1)
        weight = self._weight
        ce = F.nll_loss(log_probs, target, weight=weight, reduction="mean")

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=num_classes).permute(0, -1, *range(1, target.dim()))
        one_hot = one_hot.to(device=logits.device, dtype=probs.dtype)

        reduce_dims = (0,) + tuple(range(2, logits.dim()))
        intersection = torch.sum(probs * one_hot, dim=reduce_dims)
        denominator = torch.sum(probs + one_hot, dim=reduce_dims)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice = dice.mean()

        return self.lambda_ce * ce + self.lambda_dice * dice


class DynamicBootstrappedDiceCELoss(torch.nn.Module):
    """Dice + cross-entropy loss with dynamic bootstrapping.

    The loss mixes the ground-truth labels with the model predictions based on
    class-dependent trust scores and a training-progress dependent schedule.

    Args:
        num_classes: Number of segmentation classes (including background).
        dice_weight: Weight for the Dice component. Cross-entropy is weighted
            with ``1 - dice_weight`` unless ``ce_weight`` is explicitly
            provided.
        ce_weight: Optional explicit weight for the cross-entropy term.
        class_trust: Optional sequence with per-class trust in ``[0, 1]`` where
            larger values indicate higher confidence in the provided label. If
            ``None``, trust scores are generated from ``background_trust`` and
            ``min_positive_trust``.
        background_trust: Trust assigned to the background class when
            ``class_trust`` is ``None``.
        min_positive_trust: Minimal trust assigned to the lowest positive class
            when ``class_trust`` is ``None``. Higher classes receive
            linearly increasing trust up to 1.0.
        bootstrap_start: Training progress (0-1) at which bootstrapping starts
            to take effect.
        bootstrap_end: Training progress (0-1) by which full bootstrapping is
            applied.
        bootstrap_initial_factor: Initial global bootstrapping strength.
        bootstrap_final_factor: Final global bootstrapping strength.
        schedule_power: Exponent controlling how quickly the bootstrapping
            factor increases between ``bootstrap_start`` and ``bootstrap_end``.
        smooth: Numerical stability constant for Dice computations.
        exclude_background_from_dice: If ``True`` the Dice component ignores
            the background class.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: Optional[float] = None,
        class_trust: Optional[Sequence[float]] = None,
        background_trust: float = 0.9,
        min_positive_trust: float = 0.3,
        bootstrap_start: float = 0.0,
        bootstrap_end: float = 1.0,
        bootstrap_initial_factor: float = 0.0,
        bootstrap_final_factor: float = 1.0,
        schedule_power: float = 1.0,
        smooth: float = 1e-6,
        exclude_background_from_dice: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.num_classes = num_classes
        self.dice_weight = float(dice_weight)
        self.ce_weight = 1.0 - self.dice_weight if ce_weight is None else float(ce_weight)
        self.smooth = float(smooth)
        self.exclude_background_from_dice = exclude_background_from_dice

        if class_trust is not None:
            trust = torch.tensor(class_trust, dtype=torch.float32)
            if trust.numel() != num_classes:
                raise ValueError(
                    "class_trust must have length equal to num_classes"
                )
        else:
            positive = torch.linspace(
                min_positive_trust,
                1.0,
                steps=num_classes - 1,
                dtype=torch.float32,
            )
            background = torch.tensor([background_trust], dtype=torch.float32)
            trust = torch.cat((background, positive))
        if torch.any((trust < 0) | (trust > 1)):
            mn, mx = trust.min().item(), trust.max().item()
            raise ValueError(f"class_trust entries must be in [0,1]; got min={mn:.3f}, max={mx:.3f}")
        trust = trust.clamp(0.0, 1.0)
        print(f"Dynamic boostrapping class trust: {trust.tolist()}")
        self.register_buffer("class_trust", trust)

        self.bootstrap_start = float(bootstrap_start)
        self.bootstrap_end = float(bootstrap_end)
        self.bootstrap_initial_factor = float(bootstrap_initial_factor)
        self.bootstrap_final_factor = float(bootstrap_final_factor)
        self.schedule_power = float(schedule_power)
        self.progress = 0.0

    def update_progress(self, progress: float) -> None:
        """Update the internal training progress indicator.

        Args:
            progress: Normalised training progress in ``[0, 1]``.
        """

        self.progress = float(max(0.0, min(1.0, progress)))

    def _current_bootstrap_factor(self) -> float:
        if self.bootstrap_end <= self.bootstrap_start:
            return self.bootstrap_final_factor

        if self.progress <= self.bootstrap_start:
            return self.bootstrap_initial_factor
        if self.progress >= self.bootstrap_end:
            return self.bootstrap_final_factor

        span = self.bootstrap_end - self.bootstrap_start
        relative = (self.progress - self.bootstrap_start) / span
        relative = relative ** self.schedule_power
        return (
            self.bootstrap_initial_factor
            + relative * (self.bootstrap_final_factor - self.bootstrap_initial_factor)
        )

    def _prepare_target(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if target.dim() == 1:
            raise ValueError("Target must have at least 2 dimensions")

        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)

        if target.dtype != torch.long:
            target = target.long()

        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, -1, *range(1, target.dim()))
        one_hot = one_hot.to(dtype=torch.float32)
        return target, one_hot

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("Logits tensor must have at least 2 dimensions")

        target, one_hot = self._prepare_target(target)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)

        bootstrap_factor = self._current_bootstrap_factor()
        class_trust = self.class_trust.to(device=logits.device)
        pixel_trust = class_trust.gather(0, target.view(-1)).view_as(target)
        lambda_map = (1.0 - pixel_trust) * bootstrap_factor
        lambda_map = lambda_map.clamp(0.0, 1.0)
        lambda_map = lambda_map.unsqueeze(1)

        probs = probs.detach()

        mixed_target = (1.0 - lambda_map) * one_hot.to(device=logits.device) + lambda_map * probs

        ce_loss = -(mixed_target * log_probs).sum(dim=1).mean()

        dice_loss = torch.tensor(0.0, device=logits.device)
        if self.dice_weight > 0:
            dims = (0,) + tuple(range(2, logits.dim()))
            probs_flat = probs
            one_hot_flat = one_hot.to(device=logits.device)
            intersection = torch.sum(probs_flat * one_hot_flat, dim=dims)
            denominator = torch.sum(probs_flat + one_hot_flat, dim=dims)
            dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

            if self.exclude_background_from_dice:
                dice_per_class = dice_per_class[1:]
                trust_weights = class_trust[1:]
            else:
                trust_weights = class_trust

            weight_sum = torch.clamp(trust_weights.sum(), min=self.smooth)
            dice_loss = (dice_per_class * trust_weights).sum() / weight_sum

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return loss


def loss_getter(
    name: str,
    class_weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    dice_w: float = 0.5,
    **kwargs,
) -> torch.nn.Module:
    if class_weight is not None and not isinstance(class_weight, torch.Tensor):
        class_weight = torch.tensor(class_weight, dtype=torch.float32)

    if name == "dice_ce":
        try:
            from monai.losses.dice import DiceCELoss  # type: ignore
            print(f"Using Monai Dice Loss. Using class weights: {class_weight}")

            return DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                weight=class_weight,
                lambda_dice=dice_w,
                lambda_ce=1.0 - dice_w,
            )
        except ImportError:
            return SimpleDiceCELoss(
                weight=class_weight,
                lambda_dice=dice_w,
                lambda_ce=1.0 - dice_w,
            )
    if name == "dynamic_bootstrap":
        if "num_classes" not in kwargs:
            raise ValueError("num_classes must be provided for dynamic_bootstrap loss")
        if class_weight is not None:
            kwargs.setdefault("class_trust", class_weight.tolist())
        return DynamicBootstrappedDiceCELoss(dice_weight=dice_w, **kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {name}")
