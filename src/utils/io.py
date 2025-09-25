import os
from typing import Optional, Sequence

import cv2
import numpy as np


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

def find_by_basename(directory: str, basename: str, exts: Sequence[str]) -> Optional[str]:
    """
    Return the first existing file path matching `<basename>.<ext>`.

    Args:
        directory: folder to search
        basename: file stem without extension
        exts: candidate extensions without dot, ordered by preference
    """
    for ext in exts:
        candidate = os.path.join(directory, f"{basename}.{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None

def read_image(path: str, strict: bool = True) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        if not strict:
            return None
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_mask(path: str, strict: bool = True) -> Optional[np.ndarray]:
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        if not strict:
            return None
        raise RuntimeError(f"Failed to read mask: {path}")
    return mask