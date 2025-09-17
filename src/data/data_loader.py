import os

import albumentations as alb
import cv2
import lightning as pl
import numpy as np
import torch
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate


# -------------------------------- #
#   Helper: Albumentations wrapper
# -------------------------------- #
def alb_transform_wrapper(alb_transform):
    def _transform_fn(pil_img, pil_mask):
        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        np_mask = np.array(pil_mask)
        augmented = alb_transform(image=np_img, mask=np_mask)
        return augmented["image"], augmented["mask"]

    return _transform_fn


# -------------------------------- #
#   Dataset Wrapper
# -------------------------------- #
class PLDatasetWrapper(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform  # This should be a function(img, mask) -> (img, mask)
        # self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        # return len(self.images)
        return len(self.masks)

    def __getitem__(self, idx):
        # img_name = self.images[idx]
        mask_name = self.masks[idx]
        img_name = os.path.splitext(mask_name)[0] + ".jpg"
        # Remove ext, then add .png (this is done so that even if images are saved in .jpg masks
        # can still be found in .png format)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure mask is resized to match image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert 255 to 1 for binary segmentation
        mask[mask == 255] = 1

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = TF.to_tensor(image, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, img_name


def _collate_fn(batch):
    imgs, masks, names = zip(*batch)
    imgs = default_collate(imgs)
    masks = default_collate(masks)
    return imgs, masks, list(names)


# -------------------------------- #
#   DataModule Wrapper
# -------------------------------- #
class DataModuleWrapper(pl.LightningDataModule):
    def __init__(
        self,
        image_dir,
        mask_dir,
        batch_size=8,
        val_split=0.2,
        height: int = 512,
        width: int = 512,
        augment: bool = True,
        num_workers: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.height = height
        self.width = width
        self.augment = augment
        self.num_workers = num_workers

        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: str = None):
        # ----- Albumentations for validation -----
        val_alb_transform = alb.Compose(
            [
                alb.Resize(self.height, self.width, p=1.0),
                ToTensorV2(),
            ]
        )

        # ----- Albumentations for training (if self.augment=True) -----
        if self.augment:
            train_alb_transform = alb.Compose(
                [
                    alb.ShiftScaleRotate(
                        shift_limit=0.4,
                        scale_limit=0.2,
                        rotate_limit=45,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        p=1.0,
                    ),
                    alb.HorizontalFlip(p=0.5),
                    alb.VerticalFlip(p=0.5),
                    alb.RandomRotate90(p=0.5),
                    alb.Resize(self.height, self.width, p=1.0),
                    ToTensorV2(),
                ]
            )
        else:
            train_alb_transform = val_alb_transform

        # Convert these Albumentations pipelines into functions
        train_transform_fn = alb_transform_wrapper(train_alb_transform)
        val_transform_fn = alb_transform_wrapper(val_alb_transform)

        train_dataset = PLDatasetWrapper(
            self.image_dir, self.mask_dir, transform=train_transform_fn
        )

        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset))), test_size=self.val_split, random_state=42
        )

        val_dataset = PLDatasetWrapper(self.image_dir, self.mask_dir, transform=val_transform_fn)

        self._train_dataset = Subset(train_dataset, train_indices)
        self._val_dataset = Subset(val_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    # TODO: change to actual test dataset?
    def test_dataloader(self):
        return self.val_dataloader()
