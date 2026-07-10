"""Build train/val/test splits for the SIIM-ACR PNG dataset.

Takes mostly pneumothorax-positive cases (per project plan) plus a
configurable amount of negative cases for balance, then does an 80/10/10
stratified split (positive ratio preserved across splits) with a fixed
seed for reproducibility.

Usage:
    python dataset_split.py \
        --masks-dir ../data/siim_acr/masks \
        --out-dir ../data/siim_acr/splits \
        --neg-ratio 1.0
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42


def is_positive(mask_path: Path) -> bool:
    arr = np.array(Image.open(mask_path))
    return bool(arr.any())


def stratified_split(ids: list[str], val_frac: float, test_frac: float, rng: random.Random):
    ids = ids[:]
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = ids[:n_val]
    test = ids[n_val:n_val + n_test]
    train = ids[n_val + n_test:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                     help="negatives kept = neg_ratio * positives")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    positives, negatives = [], []
    for mask_path in sorted(args.masks_dir.glob("*.png")):
        image_id = mask_path.stem
        if is_positive(mask_path):
            positives.append(image_id)
        else:
            negatives.append(image_id)

    rng.shuffle(negatives)
    n_neg_keep = min(len(negatives), int(len(positives) * args.neg_ratio))
    kept_negatives = negatives[:n_neg_keep]

    pos_train, pos_val, pos_test = stratified_split(positives, args.val_frac, args.test_frac, rng)
    neg_train, neg_val, neg_test = stratified_split(kept_negatives, args.val_frac, args.test_frac, rng)

    splits = {
        "train": sorted(pos_train + neg_train),
        "val": sorted(pos_val + neg_val),
        "test": sorted(pos_test + neg_test),
    }

    for name, ids in splits.items():
        out_path = args.out_dir / f"{name}.txt"
        out_path.write_text("\n".join(ids) + "\n")

    summary = {
        "total_positive": len(positives),
        "total_negative_available": len(negatives),
        "negative_kept": len(kept_negatives),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
