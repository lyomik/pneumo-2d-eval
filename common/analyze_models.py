"""Characterise how each model fails, not just how well it scores.

Dice alone hides the mechanism: a model can lose Dice by drawing too much
or by drawing too little, and those two failure modes need opposite
handling downstream. This reports, per model:

  recall     pixel-level share of GT foreground recovered
  precision  share of predicted foreground that is correct
             (low precision = over-segmentation)
  area ratio median(predicted area / GT area) over positives it fired on
             (>1 = draws bigger than the lesion)
  FP rate    negatives where the model claims a lesion
  silence    positives where the model outputs nothing at all

Usage:
    python analyze_models.py [--split test]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"
MODELS = {
    "nnU-Net": ROOT / "outputs/nnunet_preds",
    "MedSAM": ROOT / "outputs/medsam_preds",
    "Swin-UNet": ROOT / "outputs/swinunet_preds",
}


def load(path: Path) -> np.ndarray:
    return np.array(Image.open(path)) > 0


def analyse(preds_dir: Path, ids: list[str]) -> dict:
    tp = fp = fn = 0
    area_ratios = []
    fp_negatives = n_negatives = 0
    silent = n_positives = 0

    for image_id in ids:
        gt = load(MASKS / f"{image_id}.png")
        pred = load(preds_dir / f"{image_id}.png")

        if gt.any():
            n_positives += 1
            tp += np.logical_and(pred, gt).sum()
            fn += np.logical_and(~pred, gt).sum()
            fp += np.logical_and(pred, ~gt).sum()
            if pred.any():
                area_ratios.append(pred.sum() / gt.sum())
            else:
                silent += 1
        else:
            n_negatives += 1
            if pred.any():
                fp_negatives += 1

    return {
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp),
        "area_ratio": float(np.median(area_ratios)),
        "fp_rate": fp_negatives / n_negatives,
        "silence": silent / n_positives,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ids = [x for x in (SPLITS / f"{args.split}.txt").read_text().splitlines() if x]

    header = f"{'model':10s} {'recall':>7s} {'prec':>7s} {'area/GT':>8s} {'FP rate':>8s} {'silence':>8s}"
    print(header)
    print("-" * len(header))
    for name, preds_dir in MODELS.items():
        if not preds_dir.exists():
            print(f"{name:10s} (no predictions)")
            continue
        m = analyse(preds_dir, ids)
        print(f"{name:10s} {m['recall']:7.3f} {m['precision']:7.3f} "
              f"{m['area_ratio']:8.2f} {m['fp_rate']:7.1%} {m['silence']:7.1%}")


if __name__ == "__main__":
    main()
