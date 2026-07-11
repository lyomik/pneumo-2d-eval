"""PNG chest-X-ray segmentation dataset shared by the from-scratch models.

Reads split id lists and returns (image, mask) at a requested square size.
Images are single-channel float in [0,1]; masks are int64 {0,1}. Light
augmentation (h-flip + small rotation) is applied only when augment=True.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/siim_acr/images"
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"


def read_ids(split: str) -> list[str]:
    return [x for x in (SPLITS / f"{split}.txt").read_text().splitlines() if x]


class SiimSegDataset(Dataset):
    def __init__(self, split: str, img_size: int = 224, augment: bool = False):
        self.ids = read_ids(split)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _load(self, image_id: str):
        img = Image.open(IMAGES / f"{image_id}.png").convert("L")
        msk = Image.open(MASKS / f"{image_id}.png").convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk, dtype=np.uint8) > 127).astype(np.int64)
        return img, msk

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img, msk = self._load(image_id)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                msk = np.ascontiguousarray(msk[:, ::-1])

        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        msk_t = torch.from_numpy(msk)               # (H, W)
        return {"image": img_t, "label": msk_t, "id": image_id}
