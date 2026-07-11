"""Render GT vs. three-model prediction overlays for test cases.

Each output page holds --per-page cases, one per row, with four columns:
ground truth, nnU-Net, MedSAM, Swin-UNet. Per-case Dice is drawn on each
prediction tile.

Case selection matches the first verification figure: pneumothorax-positive
cases with a non-trivial GT area where nnU-Net predicted something. Use
--skip to continue past cases already rendered.

Usage:
    python make_overlays.py --skip 4 --count 12 --per-page 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
from metrics import dice_score  # noqa: E402

IMAGES = ROOT / "data/siim_acr/images"
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"
PREDS = {
    "nnU-Net (red)": (ROOT / "outputs/nnunet_preds", (255, 60, 60)),
    "MedSAM (blue)": (ROOT / "outputs/medsam_preds", (60, 140, 255)),
    "Swin-UNet (yellow)": (ROOT / "outputs/swinunet_preds", (255, 200, 0)),
}
TILE = 300


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path)) > 0


def overlay(base: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    rgb = np.stack([base] * 3, -1).astype(np.uint8)
    rgb[mask > 0] = color
    return rgb


def select_cases(skip: int, count: int, min_area: int) -> list[str]:
    ids = [x for x in (SPLITS / "test.txt").read_text().splitlines() if x]
    picked = []
    for image_id in ids:
        gt = load_mask(MASKS / f"{image_id}.png")
        nn = load_mask(PREDS["nnU-Net (red)"][0] / f"{image_id}.png")
        if gt.sum() > min_area and nn.sum() > 0:
            picked.append(image_id)
        if len(picked) >= skip + count:
            break
    return picked[skip:skip + count]


def render_page(cases: list[str], out_path: Path):
    rows, dices = [], []
    for image_id in cases:
        img = np.array(Image.open(IMAGES / f"{image_id}.png").convert("L"))
        gt = load_mask(MASKS / f"{image_id}.png")

        tiles = [overlay(img, gt, (0, 255, 0))]
        row_dice = []
        for _, (pred_dir, color) in PREDS.items():
            pred = load_mask(pred_dir / f"{image_id}.png")
            tiles.append(overlay(img, pred, color))
            row_dice.append(dice_score(pred, gt))

        tiles = [np.array(Image.fromarray(t).resize((TILE, TILE))) for t in tiles]
        rows.append(np.concatenate(tiles, axis=1))
        dices.append(row_dice)

    grid = np.concatenate(rows, axis=0)
    canvas = Image.new("RGB", (TILE * 4, grid.shape[0] + 40), (20, 20, 20))
    canvas.paste(Image.fromarray(grid), (0, 40))

    draw = ImageDraw.Draw(canvas)
    headers = ["GT (green)"] + list(PREDS.keys())
    for col, title in enumerate(headers):
        draw.text((col * TILE + 8, 14), title, fill=(255, 255, 255))
    colors = [c for _, c in PREDS.values()]
    for r, row_dice in enumerate(dices):
        for c, d in enumerate(row_dice):
            draw.text(((c + 1) * TILE + 8, 44 + r * TILE), f"{d:.3f}", fill=colors[c])

    canvas.save(out_path)
    return dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--per-page", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=3000)
    ap.add_argument("--start-index", type=int, default=2, help="page number to start naming at")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "verification"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(args.skip, args.count, args.min_area)
    print(f"selected {len(cases)} cases (skipped first {args.skip})")

    for page, start in enumerate(range(0, len(cases), args.per_page)):
        chunk = cases[start:start + args.per_page]
        out_path = out_dir / f"test_overlays_{args.start_index + page}.png"
        dices = render_page(chunk, out_path)
        print(f"\n{out_path.name}")
        for image_id, (dn, dm, ds) in zip(chunk, dices):
            print(f"  ...{image_id[-12:]}  nnU-Net {dn:.3f}  MedSAM {dm:.3f}  Swin {ds:.3f}")


if __name__ == "__main__":
    main()
