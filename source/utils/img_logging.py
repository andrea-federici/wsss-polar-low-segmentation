import torch
from torchvision.utils import make_grid


VOC_PALETTE = torch.tensor([
    [  0,   0,   0],   # class 0: background
    [128,   0,   0],   # class 1
    [  0, 128,   0],   # class 2
    [128, 128,   0],   # class 3
    [  0,   0, 128],   # class 4
    [128,   0, 128],   # class 5
    [  0, 128, 128],   # class 6
    [128, 128, 128],   # class 7
    [ 64,   0,   0],   # class 8
    [192,   0,   0],   # class 9
    [ 64, 128,   0],   # class 10
    [192, 128,   0],   # class 11
    [ 64,   0, 128],   # class 12
    [192,   0, 128],   # class 13
    [ 64, 128, 128],   # class 14
    [192, 128, 128],   # class 15
    [  0,  64,   0],   # class 16
    [128,  64,   0],   # class 17
    [  0, 192,   0],   # class 18
    [128, 192,   0],   # class 19
    [  0,  64, 128],   # class 20
], dtype=torch.uint8)


def colorize_mask(mask: torch.Tensor, palette: torch.Tensor=VOC_PALETTE) -> torch.Tensor:
    """
    Convert a [H,W] mask of class indices (0..C-1) into a 3-channel color image.
    Any values >= palette.shape[0] (e.g. 255) will be clamped or set to 0 by default.
    Args:
        mask: shape [H,W], with integer labels in [0..20] for VOC
        palette: shape [C,3], each row an [R,G,B].
    Returns:
        colored: shape [3,H,W], an RGB image (uint8).
    """
    # Ensure mask is on CPU and long:
    mask = mask.long().cpu()

    mask = torch.where(mask < palette.shape[0], mask, torch.zeros_like(mask))

    # Map each label to [R,G,B]
    # result = [H,W,3]
    colored = palette[mask]

    # Convert shape to [3,H,W] and keep it uint8
    colored = colored.permute(2, 0, 1)  # [3,H,W]
    return colored


def xy_grid_voc(x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor,
                max_samples=4) -> torch.Tensor:
    """
    Create a single visualization combining:
      - The input image (assumed to be 3-channel, shape [B,3,H,W])
      - The ground-truth mask, colorized
      - The predicted mask, colorized
    We do this only for the first few samples in the batch.

    Returns a big image with shape [3, H*(num_samples), W*3].
    """
    # We'll limit ourselves to a few samples, e.g. up to 4
    num_samples = min(x.shape[0], max_samples)

    # If y_pred has shape [B,C,H,W], get the argmax
    if y_pred.dim() == 4 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(dim=1)

    # We'll accumulate each sample's triple (image, GT, pred) side-by-side
    visualization_list = []
    for i in range(num_samples):
        rgb_img = x[i]              # shape [3,H,W], presumably float in [0,1]
        gt_mask = y[i]             # shape [H,W]
        pred_mask = y_pred[i]      # shape [H,W]

        # Colorize the ground-truth and prediction
        gt_colored = colorize_mask(gt_mask)      # [3,H,W], uint8
        pred_colored = colorize_mask(pred_mask)  # [3,H,W], uint8

        # We assume `rgb_img` might be float, while gt_colored is uint8 in [0..255].
        # Convert them to the same type & scale for concatenation:
        if rgb_img.dtype != torch.uint8:
            # We'll scale the RGB image from [0,1] float to [0,255] uint8 for visual consistency
            rgb_img = (rgb_img * 255.0).clamp(0,255).byte()

        # Now we cat them side-by-side -> [3, H, W*3]
        combined = torch.cat([rgb_img.cpu(), gt_colored, pred_colored], dim=2)
        visualization_list.append(combined)

    # Finally, stack the samples vertically -> shape [3, H*num_samples, W*3]
    final = torch.cat(visualization_list, dim=1)

    return final

def xy_grid(x, y, y_pred):

    xy = torch.concat([x, y.unsqueeze(1)], axis=1)
    y_pred_thresh = torch.argmax(y_pred, dim=1, keepdim=True)
    xy = torch.concat([xy, y_pred_thresh], axis=1)

    grid = make_grid(xy, pad_value=.5, nrow=16)
    img = torch.concat([grid[i:i+1,:,:] for i in range(grid.shape[0])], axis=1)

    return img