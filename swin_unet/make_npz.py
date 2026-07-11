"""Convert the SIIM-ACR PNG splits into the format the upstream Swin-UNet
train.py expects, so we can run the official training script unchanged.

Produces:
    <out>/<case>.npz                # arrays: image (HxW float32 [0,1]),
                                     #         label (HxW uint8 {0,1})
    repo/lists/pneumo/train.txt      # one case name per line
    repo/lists/pneumo/val.txt

train.py only appends "train_npz" to --root_path when --dataset is the
built-in "Synapse"; for our "pneumo" dataset it reads npz straight from
--root_path, so files go directly in <out>. Case lists live in
./lists/<dataset>/. Images are stored at the model input size (224) so the
trainer's RandomGenerator only augments, without resizing.

Usage:
    python make_npz.py --img-size 224
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "swin_unet" / "repo"
IMAGES = ROOT / "data/siim_acr/images"
MASKS = ROOT / "data/siim_acr/masks"
SPLITS = ROOT / "data/siim_acr/splits"


def read_ids(split: str) -> list[str]:
    return [x for x in (SPLITS / f"{split}.txt").read_text().splitlines() if x]


def convert_case(image_id: str, img_size: int):
    img = Image.open(IMAGES / f"{image_id}.png").convert("L").resize(
        (img_size, img_size), Image.BILINEAR)
    msk = Image.open(MASKS / f"{image_id}.png").convert("L").resize(
        (img_size, img_size), Image.NEAREST)
    image = np.asarray(img, dtype=np.float32) / 255.0
    label = (np.asarray(msk, dtype=np.uint8) > 127).astype(np.uint8)
    return image, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--dataset-name", type=str, default="pneumo")
    ap.add_argument("--out", type=str, default=str(ROOT / "swin_unet" / "npz_data"))
    args = ap.parse_args()

    npz_dir = Path(args.out) / "train_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    list_dir = REPO / "lists" / args.dataset_name
    list_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        ids = read_ids(split)
        for image_id in ids:
            image, label = convert_case(image_id, args.img_size)
            np.savez_compressed(npz_dir / f"{image_id}.npz", image=image, label=label)
        (list_dir / f"{split}.txt").write_text("\n".join(ids) + "\n")
        print(f"{split}: {len(ids)} cases -> {npz_dir} ; list -> {list_dir}/{split}.txt")

    print(f"root_path to pass train.py: {args.out}")
    print(f"list_dir will be: ./lists/{args.dataset_name}")


if __name__ == "__main__":
    main()
