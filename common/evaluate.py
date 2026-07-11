"""Evaluate a model's predicted masks against ground truth on the test split.

Computes mean Dice over all test cases, and separately over the
pneumothorax-positive subset (where the task is actually hard — negatives
score 1.0 whenever a model correctly predicts nothing, which inflates the
overall mean). Predictions and GT may be 0/1 or 0/255; both are thresholded
at >0.

Usage:
    python evaluate.py --preds-dir ../outputs/nnunet_preds --name nnU-Net
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"

import sys
sys.path.insert(0, str(ROOT / "common"))
from metrics import dice_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    ids = [x for x in (SPLITS / f"{args.split}.txt").read_text().splitlines() if x]

    all_dice, pos_dice = [], []
    missing = 0
    for image_id in ids:
        pred_path = preds_dir / f"{image_id}.png"
        if not pred_path.exists():
            missing += 1
            continue
        gt = np.array(Image.open(MASKS / f"{image_id}.png")) > 0
        pred = np.array(Image.open(pred_path)) > 0
        d = dice_score(pred, gt)
        all_dice.append(d)
        if gt.any():
            pos_dice.append(d)

    print(f"=== {args.name} ({args.split}) ===")
    print(f"  cases evaluated:      {len(all_dice)} (missing preds: {missing})")
    print(f"  mean Dice (all):      {np.mean(all_dice):.4f}")
    print(f"  mean Dice (positives):{np.mean(pos_dice):.4f}  (n={len(pos_dice)})")
    return float(np.mean(all_dice)), float(np.mean(pos_dice))


if __name__ == "__main__":
    main()
