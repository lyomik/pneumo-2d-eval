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
    """Map ImageId -> list of RLE strings.

    stage_2_train.csv carries an unnamed leading index column, so columns are
    resolved by header name rather than position.
    """
    rle_map: dict[str, list[str]] = defaultdict(list)
    with open(rle_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"].strip()
            encoded_pixels = row["EncodedPixels"].strip()
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
    skipped_unlabeled = []

    for dcm_path in dcm_files:
        image_id = dcm_path.stem

        rles = rle_map.get(image_id)
        if rles is None:
            # No annotation row at all. An all-zero mask would assert "no
            # pneumothorax", which is a different claim from "unlabeled".
            skipped_unlabeled.append(image_id)
            continue

        ds = pydicom.dcmread(dcm_path)
        pixels = ds.pixel_array
        if pixels.dtype != np.uint8:
            raise SystemExit(f"{image_id}: expected uint8 pixels, got {pixels.dtype}")
        height, width = pixels.shape

        mask = np.zeros((height, width), dtype=np.uint8)
        for rle in rles:
            mask |= rle2mask(rle, width, height)
        if mask.any():
            n_positive += 1

        Image.fromarray(pixels).save(args.out_images / f"{image_id}.png")
        Image.fromarray(mask * 255).save(args.out_masks / f"{image_id}.png")
        n_converted += 1

    print(f"Converted {n_converted} labeled images -> {args.out_images}")
    print(f"Positive (pneumothorax): {n_positive}  Negative: {n_converted - n_positive}")
    print(f"Skipped (DICOM present but no RLE row): {len(skipped_unlabeled)}")
    if skipped_unlabeled:
        print("  e.g.", skipped_unlabeled[:3])


if __name__ == "__main__":
    main()
