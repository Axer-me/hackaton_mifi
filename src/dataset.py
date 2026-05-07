import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int, mode: str) -> A.Compose:
    # albumentations >= 2.0: size args changed to keyword tuples
    resize_to = int(img_size * 256 / 224)
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=resize_to, width=resize_to),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class RoomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: A.Compose,
        mode: str = "train",   # "train", "val", "test"
        preload: bool = False,  # load all images to RAM once
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode
        self.cache: dict = {}

        if preload:
            print(f"  Preloading {len(self.df)} images to RAM...", flush=True)
            for i, row in self.df.iterrows():
                img_path = os.path.join(images_dir, f"{row['image_id_ext']}.jpg")
                try:
                    self.cache[i] = np.array(Image.open(img_path).convert("RGB"))
                except Exception:
                    self.cache[i] = np.zeros((224, 224, 3), dtype=np.uint8)
            print("  Preload done.", flush=True)

    def _load_image(self, idx: int, img_id) -> np.ndarray:
        if idx in self.cache:
            return self.cache[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        try:
            return np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._load_image(idx, row["image_id_ext"])
        tensor = self.transform(image=image)["image"]

        if self.mode == "test":
            return tensor, str(row["image_id_ext"])
        return tensor, int(row["result"])


def build_weighted_sampler(df: pd.DataFrame):
    from torch.utils.data import WeightedRandomSampler
    counts = df["result"].value_counts().sort_index()
    # Fill missing classes with 1 to avoid division by zero
    all_classes = range(counts.index.min(), counts.index.max() + 1)
    weights_per_class = {c: 1.0 / counts.get(c, 1) for c in all_classes}
    sample_weights = df["result"].map(weights_per_class).values
    return WeightedRandomSampler(
        torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )
