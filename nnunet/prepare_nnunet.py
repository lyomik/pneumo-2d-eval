"""Convert the SIIM-ACR PNG dataset into nnU-Net v2 raw format.

train + val splits -> imagesTr/labelsTr (nnU-Net runs its own 5-fold CV).
test split         -> imagesTs (held out; GT masks kept in the PNG dir for
                                 our own Dice computation later).

Masks are remapped from 0/255 to 0/1 as nnU-Net expects consecutive integer
labels. Images are single-channel grayscale (modality 0000).

Usage:
    python prepare_nnunet.py --dataset-id 1
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data/siim_acr/images"
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"


def read_ids(name: str) -> list[str]:
    return [x for x in (SPLITS / f"{name}.txt").read_text().splitlines() if x]


def copy_image(image_id: str, dst_dir: Path):
    # nnU-Net channel suffix _0000; keep as grayscale PNG.
    shutil.copy(IMAGES / f"{image_id}.png", dst_dir / f"{image_id}_0000.png")


def write_label(image_id: str, dst_dir: Path):
    mask = np.array(Image.open(MASKS / f"{image_id}.png"))
    binary = (mask > 127).astype(np.uint8)
    Image.fromarray(binary).save(dst_dir / f"{image_id}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=1)
    args = ap.parse_args()

    raw_root = ROOT / "nnunet" / "nnUNet_raw"
    ds_name = f"Dataset{args.dataset_id:03d}_Pneumothorax"
    ds_dir = raw_root / ds_name
    for sub in ["imagesTr", "labelsTr", "imagesTs"]:
        (ds_dir / sub).mkdir(parents=True, exist_ok=True)

    train_ids = read_ids("train") + read_ids("val")
    test_ids = read_ids("test")

    for image_id in train_ids:
        copy_image(image_id, ds_dir / "imagesTr")
        write_label(image_id, ds_dir / "labelsTr")

    for image_id in test_ids:
        copy_image(image_id, ds_dir / "imagesTs")

    dataset_json = {
        "channel_names": {"0": "CXR"},
        "labels": {"background": 0, "pneumothorax": 1},
        "numTraining": len(train_ids),
        "file_ending": ".png",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"Wrote {ds_name}")
    print(f"  imagesTr/labelsTr: {len(train_ids)}")
    print(f"  imagesTs:          {len(test_ids)}")
    print(f"  -> {ds_dir}")


if __name__ == "__main__":
    main()
