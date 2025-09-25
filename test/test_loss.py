import torch
from src.utils.seg_losses import loss_getter


def test_dice_ce_loss_runs():
    B, C, H, W = 2, 3, 16, 16
    x = torch.randn(B, C, H, W, requires_grad=True)
    y = torch.randint(0, C, (B, H, W))  # Targets in [0, C-1], shape (B, H, W)

    loss_fn = loss_getter(name="dice_ce", class_weight=None)
    loss = loss_fn(x, y)

    # Assert loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    # Check that backward works
    loss.backward()
    assert x.grad is not None


def test_dynamic_bootstrap_loss_runs():
    B, C, H, W = 2, 4, 8, 8
    x = torch.randn(B, C, H, W, requires_grad=True)
    y = torch.randint(0, C, (B, H, W))

    loss_fn = loss_getter(
        name="dynamic_bootstrap",
        num_classes=C,
        dice_w=0.4,
        background_trust=0.7,
        min_positive_trust=0.3,
        bootstrap_start=0.0,
        bootstrap_end=1.0,
        bootstrap_initial_factor=0.0,
        bootstrap_final_factor=1.0,
    )
    loss_fn.update_progress(0.5)
    loss = loss_fn(x, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    loss.backward()
    assert x.grad is not None
