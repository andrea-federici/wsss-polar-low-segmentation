import os
import random
from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.functional as TF

import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np

# -------------------------------- #
#   Helper: Albumentations wrapper
# -------------------------------- #
def alb_transform_wrapper(alb_transform):
    """
    Returns a function that:
      1) Converts PIL images/masks to NumPy arrays,
      2) Runs them through an Albumentations transform,
      3) Returns the (image, mask) as tensors.
    """
    def _transform_fn(pil_img, pil_mask):
        # Convert PIL -> NumPy
        np_img = np.array(pil_img)
        np_mask = np.array(pil_mask)
        # Run Albumentations
        augmented = alb_transform(image=np_img, mask=np_mask)
        # Extract results
        aug_img, aug_mask = augmented["image"], augmented["mask"]
        return aug_img, aug_mask
    return _transform_fn


# -------------------------------- #
#   VOCDatasetWrapper
# -------------------------------- #
class VOCDatasetWrapper(Dataset):
    """
    Wraps torchvision.datasets.VOCSegmentation so we can easily apply the same
    transform to both the image and the mask (Albumentations included).
    """
    def __init__(self, root, year="2012", image_set="train", download=False,
                 transform=None):
        super().__init__()
        self.voc = VOCSegmentation(root=root,
                                   year=year,
                                   image_set=image_set,
                                   download=download)
        self.transform = transform  # This should be a function(img, mask) -> (img, mask)

    def __getitem__(self, idx):
        # Returns PIL Image and PIL mask
        pil_img, pil_mask = self.voc[idx]
        if self.transform is not None:
            img, mask = self.transform(pil_img, pil_mask)  # Albumentations call
        else:
            # Fallback: just convert to standard Torch tensor
            img = TF.to_tensor(pil_img)
            mask = torch.as_tensor(np.array(pil_mask), dtype=torch.long)
        return img, mask

    def __len__(self):
        return len(self.voc)

# -------------------------------- #
#   VOCDataModule
# -------------------------------- #
class VOCDataModule(pl.LightningDataModule):
    """
    A drop-in replacement for your old AvalancheDataModule that
    uses Pascal VOC, but with the same Albumentations transforms
    from the old code's get_train_loader, get_valid_loader, get_test_loader.
    """

    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 4,
        height: int = 512,
        width: int = 512,
        augment: bool = True,
        num_workers: int = 2,
        *args,
        **kwargs
    ):
        """
        Args:
            data_dir: Where VOC2012 data is stored (or will be downloaded).
            batch_size: Dataloader batch size.
            height, width: The spatial size to which images are cropped/padded.
            augment: Whether to apply the heavy data augmentation pipeline for training.
            num_workers: Number of DataLoader workers.
        """
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.augment = augment
        self.num_workers = num_workers

        # We will define the Albumentations transforms in setup() or on the fly.
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def prepare_data(self):
        """
        Download the dataset. Called once per machine in a multi-GPU setting.
        """
        # This ensures Pascal VOC is downloaded. We only need "train" and "val".
        VOCSegmentation(root=self.data_dir, year='2012', image_set='train', download=True)
        VOCSegmentation(root=self.data_dir, year='2012', image_set='val', download=True)

    def setup(self, stage: Optional[str] = None):
        """
        Create train/val/test Datasets. Called on every GPU in DDP.
        """
        # ----- Albumentations for training (if self.augment=True) -----
        if self.augment:
            # Mirror of old "get_train_loader(..., augment=True)"
            train_alb_transform = alb.Compose([
                alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1,
                                     border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.PadIfNeeded(min_height=self.height, min_width=self.width,
                                border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                alb.RandomCrop(self.height, self.width, p=1.0),
                ToTensorV2(),
            ])
        else:
            # Mirror of old "get_train_loader(..., augment=False)"
            train_alb_transform = alb.Compose([
                alb.PadIfNeeded(min_height=self.height, min_width=self.width,
                                border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                alb.CenterCrop(self.height, self.width, p=1.0),
                ToTensorV2(),
            ])

        # ----- Albumentations for validation -----
        # Mirror of old "get_valid_loader(...)"
        val_alb_transform = alb.Compose([
            alb.PadIfNeeded(min_height=self.height, min_width=self.width,
                            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            alb.CenterCrop(self.height, self.width, p=1.0),
            ToTensorV2(),
        ])

        # ----- Albumentations for test -----
        # Mirror of old "get_test_loader(...)"
        test_alb_transform = alb.Compose([
            alb.PadIfNeeded(min_height=self.height, min_width=self.width,
                            border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            alb.CenterCrop(self.height, self.width, p=1.0),
            ToTensorV2(),
        ])

        # Convert these Albumentations pipelines into functions
        train_transform_fn = alb_transform_wrapper(train_alb_transform)
        val_transform_fn   = alb_transform_wrapper(val_alb_transform)
        test_transform_fn  = alb_transform_wrapper(test_alb_transform)

        # Setup for train/val
        if stage == "fit" or stage is None:
            self._train_dataset = VOCDatasetWrapper(
                root=self.data_dir,
                year="2012",
                image_set="train",
                download=False,
                transform=train_transform_fn,
            )
            self._val_dataset = VOCDatasetWrapper(
                root=self.data_dir,
                year="2012",
                image_set="val",
                download=False,
                transform=val_transform_fn,
            )

        # Setup for test (reuse "val" as test because VOC does not provide test labels)
        if stage == "test" or stage is None:
            self._test_dataset = VOCDatasetWrapper(
                root=self.data_dir,
                year="2012",
                image_set="val",
                download=False,
                transform=test_transform_fn,
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # as in old code
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        """
        Optional, if you had a predict_dataloader in the old code (like 'disp_loader').
        If you don't need it, you can omit this method.
        """
        return self.test_dataloader()