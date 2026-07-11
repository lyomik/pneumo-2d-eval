"""Run a trained Swin-UNet on the SIIM-ACR test split.

Loads the checkpoint saved by the official train.py, predicts at 224x224,
upsamples the predicted mask back to the native 1024x1024, and writes a
binary PNG (0/255) per case into outputs/swinunet_preds/.

Usage:
    python infer_swin.py --checkpoint <path to best_model.pth>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "swin_unet" / "repo"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "common"))

from config import _C  # noqa: E402
from networks.vision_transformer import SwinUnet  # noqa: E402
from seg_dataset import read_ids  # noqa: E402

IMAGES = ROOT / "data/siim_acr/images"
NATIVE = 1024


def build_config(img_size: int):
    cfg = _C.clone()
    cfg.defrost()
    cfg.merge_from_file(str(REPO / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"))
    cfg.DATA.IMG_SIZE = img_size
    cfg.freeze()
    return cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "swinunet_preds"))
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args.img_size)
    model = SwinUnet(cfg, img_size=args.img_size, num_classes=2).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state = state.get("model", state)  # train.py saves a bare state_dict
    model.load_state_dict(state)
    model.eval()

    ids = read_ids("test")
    for image_id in ids:
        img = Image.open(IMAGES / f"{image_id}.png").convert("L").resize(
            (args.img_size, args.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        logits = model(x)
        up = F.interpolate(logits, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False)
        pred = up.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(pred * 255).save(out_dir / f"{image_id}.png")

    print(f"Wrote {len(ids)} predictions -> {out_dir}")


if __name__ == "__main__":
    main()
