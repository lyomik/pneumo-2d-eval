"""Convert SIIM-ACR Pneumothorax DICOM images + RLE mask CSV into PNG pairs.

Usage:
    python dicom_to_png.py \
        --raw-dir ../data/raw \
        --rle-csv ../data/raw/train-rle.csv \
        --out-images ../data/siim_acr/images \
        --out-masks ../data/siim_acr/masks

Expects the raw Kaggle competition dump layout, where DICOM files live
anywhere under --raw-dir (searched recursively) and are named
"<ImageId>.dcm". Multiple RLE rows for the same ImageId are unioned
(some images have more than one annotated region).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from rle_utils import rle2mask


def load_rle_map(rle_csv: Path) -> dict[str, list[str]]:
    rle_map: dict[str, list[str]] = defaultdict(list)
    with open(rle_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            image_id, encoded_pixels = row[0].strip(), row[1].strip()
            rle_map[image_id].append(encoded_pixels)
    return rle_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--rle-csv", type=Path, required=True)
    ap.add_argument("--out-images", type=Path, required=True)
    ap.add_argument("--out-masks", type=Path, required=True)
    args = ap.parse_args()

    args.out_images.mkdir(parents=True, exist_ok=True)
    args.out_masks.mkdir(parents=True, exist_ok=True)

    rle_map = load_rle_map(args.rle_csv)
    dcm_files = sorted(args.raw_dir.rglob("*.dcm"))
    if not dcm_files:
        raise SystemExit(f"No .dcm files found under {args.raw_dir}")

    n_converted = 0
    n_positive = 0
    n_missing_rle = 0

    for dcm_path in dcm_files:
        image_id = dcm_path.stem
        ds = pydicom.dcmread(dcm_path)
        pixels = ds.pixel_array
        height, width = pixels.shape

        Image.fromarray(pixels).save(args.out_images / f"{image_id}.png")

        rles = rle_map.get(image_id)
        if rles is None:
            n_missing_rle += 1
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            for rle in rles:
                mask |= rle2mask(rle, width, height)
            if mask.any():
                n_positive += 1

        Image.fromarray(mask * 255).save(args.out_masks / f"{image_id}.png")
        n_converted += 1

    print(f"Converted {n_converted} images -> {args.out_images}")
    print(f"Positive (pneumothorax) masks: {n_positive}")
    print(f"Images with no RLE entry in CSV: {n_missing_rle}")


if __name__ == "__main__":
    main()
