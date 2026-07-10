"""Compute bounding boxes from predicted masks, for use as MedSAM box prompts.

Usage:
    python mask_to_bbox.py \
        --masks-dir ../outputs/nnunet_preds \
        --out-json ../outputs/nnunet_preds/bboxes.json

Each mask may contain multiple disconnected regions; by default this
takes the single bounding box covering all foreground pixels (matches
plain binary-mask -> box conversion used to prompt MedSAM). Images with
an empty mask get bbox = null.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max, y_max]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    bboxes = {}
    for mask_path in sorted(args.masks_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path))
        bboxes[mask_path.stem] = mask_to_bbox(mask)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(bboxes, indent=2))

    n_with_box = sum(1 for v in bboxes.values() if v is not None)
    print(f"Wrote {len(bboxes)} entries ({n_with_box} with a bbox) -> {args.out_json}")


if __name__ == "__main__":
    main()
