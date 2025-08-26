import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_gaussian,
    create_pairwise_bilateral,
)


def refine_mask_with_crf(
    image: np.ndarray, mask_prob: np.ndarray, num_classes: int = 2, iterations: int = 5
) -> np.ndarray:
    """
    Refines a segmentation mask using DenseCRF.

    Args:
        image (np.ndarray): Original image (H, W, 3) in BGR or RGB format.
        mask_prob (np.ndarray): Soft mask probability map with shape (num_classes, H, W).
                                For binary segmentation, it should be a 2 x H x W array.
        num_classes (int): Number of classes in the segmentation (default is 2).
        iterations (int): Number of mean-field inference iterations.

    Returns:
        np.ndarray: Refined mask (H, W) with labels.
    """
    h, w = image.shape[:2]
    dcrf_model = dcrf.DenseCRF2D(w, h, num_classes)

    # Compute unary potentials from the softmax probabilities.
    unary = unary_from_softmax(mask_prob)
    dcrf_model.setUnaryEnergy(unary)

    # Add pairwise Gaussian potentials (spatial smoothness)
    gaussian_sxy = 3  # standard deviation for spatial kernel
    pairwise_gaussian = create_pairwise_gaussian(
        sdims=[
            gaussian_sxy,
            gaussian_sxy,
        ],  # TODO: are we passing the correct value to sdims?
        shape=image.shape[:2],
    )
    dcrf_model.addPairwiseEnergy(pairwise_gaussian, compat=3)

    # Add pairwise bilateral potentials (spatial + color)
    bilateral_sxy = 50  # spatial standard deviation
    bilateral_srgb = 13  # color standard deviation
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=[bilateral_sxy, bilateral_sxy],  # TODO: is this correct?
        schan=[bilateral_srgb, bilateral_srgb],  # TODO: is this correct?
        img=image,
        chdim=2,
    )
    dcrf_model.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Run inference
    refined_probs = dcrf_model.inference(iterations)
    refined_probs = np.array(refined_probs).reshape((num_classes, h, w))

    # Choose the class with the highest probability at each pixel
    refined_mask = np.argmax(refined_probs, axis=0)
    return refined_mask
