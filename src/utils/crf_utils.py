import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def refine_mask_with_crf(
    image: np.ndarray, 
    mask_prob: np.ndarray, 
    num_classes: int = 2,
    iterations: int = 5, 
    gaussian_sxy: int = 3, 
    bilateral_sxy: int = 50, 
    bilateral_srgb: int = 13, 
    compat_gaussian: int = 3, 
    compat_bilateral: int = 10,
) -> np.ndarray:
    # --- Validate & massage inputs ---
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if np.issubdtype(img.dtype, np.floating) else img
        if img.max() <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]

    prob = np.asarray(mask_prob)
    if prob.ndim != 3:
        raise ValueError(f"mask_prob must be 3D, got shape {prob.shape}")
    if prob.shape[0] == H and prob.shape[1] == W and prob.shape[2] == num_classes:
        prob = np.transpose(prob, (2, 0, 1))  # to (C,H,W)

    if prob.shape != (num_classes, H, W):
        raise ValueError(f"mask_prob must have shape ({num_classes}, {H}, {W}), got {prob.shape}")

    # Ensure valid probabilities and numerical stability
    prob = prob.astype(np.float32)
    prob = np.maximum(prob, 1e-7)
    prob /= np.sum(prob, axis=0, keepdims=True)

    # >>> Make both arrays C-contiguous (critical for pydensecrf) <<<
    img  = np.ascontiguousarray(img, dtype=np.uint8)       # (H,W,3), C-contiguous
    prob = np.ascontiguousarray(prob, dtype=np.float32)    # (C,H,W), C-contiguous

    # --- Build CRF ---
    d = dcrf.DenseCRF2D(W, H, num_classes)
    unary = unary_from_softmax(prob)  # expects (C, H, W) float32, C-contiguous
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=(gaussian_sxy, gaussian_sxy), compat=compat_gaussian)
    d.addPairwiseBilateral(
        sxy=(bilateral_sxy, bilateral_sxy),
        srgb=(bilateral_srgb, bilateral_srgb, bilateral_srgb),
        rgbim=img,
        compat=compat_bilateral,
    )

    Q = d.inference(iterations)
    Q = np.array(Q, dtype=np.float32).reshape(num_classes, H, W)
    return np.argmax(Q, axis=0).astype(np.int32)

