import os
from typing import Callable, List, Optional, Sequence

import albumentations as A
import cv2
import lightning as pl
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate

from src.data.augmentation import TransformFn, alb_transform_wrapper
from src.utils.io import ensure_exists, find_by_basename, read_image, read_mask
from src.utils.seed import make_worker_init_fn


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        *,
        transform: Optional[TransformFn] = None,
        image_exts: Sequence[str] = ("png", "jpg", "jpeg"),
        mask_exts: Sequence[str] = ("png", "jpg", "jpeg"),
    ) -> None:
        super().__init__()
        ensure_exists(image_dir)
        ensure_exists(mask_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_exts = image_exts
        self.mask_exts = mask_exts

        # Use image directory as source of basenames
        self._basenames: List[str] = []
        for fname in sorted(os.listdir(image_dir)):
            stem, _ = os.path.splitext(fname)
            if find_by_basename(image_dir, stem, image_exts):
                self._basenames.append(stem)
        if not self._basenames:
            raise RuntimeError(f"No images found in {image_dir} with extensions {image_exts}.")

        missing_masks = [
            stem for stem in self._basenames
            if find_by_basename(mask_dir, stem, mask_exts) is None
        ]
        if missing_masks:
            preview = ", ".join(missing_masks[:5]) + ("..." if len(missing_masks) > 5 else "")
            raise RuntimeError(f"Masks not found for {len(missing_masks)} images: {preview}")

    def __len__(self):
        return len(self._basenames)

    def __getitem__(self, idx):
        stem = self._basenames[idx]
        img_path = find_by_basename(self.image_dir, stem, self.image_exts)
        mask_path = find_by_basename(self.mask_dir, stem, self.mask_exts)

        if img_path is None:
            raise FileNotFoundError(
                f"No image found for '{stem}' in {self.image_dir} with exts {self.image_exts}."
            )
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for '{stem}' in {self.mask_dir} with exts {self.mask_exts}."
            )

        image = read_image(img_path)
        mask = read_mask(mask_path)
        assert image is not None and mask is not None

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.transform is not None:
            image_t, mask_t = self.transform(image, mask)
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask.astype(np.int64))

        return image_t, mask_t, os.path.basename(img_path)


def _collate_with_names(batch):
    imgs, masks, names = zip(*batch)
    return default_collate(imgs), default_collate(masks), list(names)


class SegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        height: int,
        width: int,
        *,
        val_split: float = 0.2,
        augment: bool = False,
        aug_list: Optional[List[A.BasicTransform | A.BaseCompose]] = None,
        batch_size: int = 8,
        num_workers: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.height = height
        self.width = width
        self.val_split = val_split
        self.augment = augment
        self.aug_list = aug_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._generator: Optional[torch.Generator] = None
        self._worker_init_fn: Optional[Callable[[int], None]] = None

        self._train_dataset: Optional[Subset] = None
        self._val_dataset: Optional[Subset] = None

    def setup(self, stage: str | None = None) -> None:
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
            self._worker_init_fn = make_worker_init_fn(self.seed)
        else:
            self._generator = None
            self._worker_init_fn = None

        base_ops = [A.Resize(self.height, self.width), ToTensorV2()]

        if self.augment:
            if self.aug_list is not None:
                aug_ops = self.aug_list
            else:
                raise ValueError("`aug_list` must be provided if `augment` is True.")
        else:
            aug_ops = []

        train_tf = A.Compose(aug_ops + base_ops)
        val_tf   = A.Compose(base_ops)

        train_ds_full = SegmentationDataset(
            self.image_dir, self.mask_dir, transform=alb_transform_wrapper(train_tf)
        )
        val_ds_full = SegmentationDataset(
            self.image_dir, self.mask_dir, transform=alb_transform_wrapper(val_tf)
        )

        indices = list(range(len(train_ds_full)))
        train_idx, val_idx = train_test_split(
            indices, test_size=self.val_split, random_state=self.seed
        )
        self._train_dataset = Subset(train_ds_full, train_idx)
        self._val_dataset = Subset(val_ds_full, val_idx)

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call `setup()` before requesting dataloaders.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_with_names,
            worker_init_fn=self._worker_init_fn,
            generator=self._generator,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call `setup()` before requesting dataloaders.")
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_with_names,
            worker_init_fn=self._worker_init_fn,
            generator=self._generator,
        )

    def test_dataloader(self):
        return self.val_dataloader()  # TODO: change if separate test set is available
