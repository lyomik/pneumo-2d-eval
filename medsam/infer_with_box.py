"""MedSAM inference on the SIIM-ACR test split, prompted by nnU-Net boxes.

For each test image we take the bounding box of nnU-Net's predicted mask
(computed by common/mask_to_bbox.py) and feed it to MedSAM as the box
prompt. Images where nnU-Net predicted nothing get an empty mask (MedSAM
needs a prompt; no box -> no prediction), which is the correct behaviour
for a negative case.

Preprocessing follows the official MedSAM_Inference.py: resize to 1024,
min-max normalize to [0,1], no ImageNet mean/std. Our images are already
1024x1024 so the box needs no rescaling, but we rescale generically in
case that changes.

Usage:
    python infer_with_box.py \
        --bboxes ../outputs/nnunet_preds/bboxes.json \
        --checkpoint repo/work_dir/MedSAM/medsam_vit_b.pth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import transform

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "medsam" / "repo"
sys.path.insert(0, str(REPO))

from segment_anything import sam_model_registry  # noqa: E402

IMAGES = ROOT / "data/siim_acr/images"


@torch.no_grad()
def medsam_infer(model, img_embed, box_1024, H, W):
    box_t = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if box_t.ndim == 2:
        box_t = box_t[:, None, :]  # (B,1,4)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    low_res, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    prob = torch.sigmoid(low_res)
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    return (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes", required=True, help="json: {case: [x0,y0,x1,y1] or null}")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "medsam_preds"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(args.device)
    model.eval()

    bboxes = json.loads(Path(args.bboxes).read_text())
    n_box = n_empty = 0

    for case, box in bboxes.items():
        img_path = IMAGES / f"{case}.png"
        if not img_path.exists():
            continue
        img = np.array(Image.open(img_path).convert("L"))
        H, W = img.shape

        if box is None:
            # nnU-Net found nothing -> no prompt -> empty prediction.
            Image.fromarray(np.zeros((H, W), np.uint8)).save(out_dir / f"{case}.png")
            n_empty += 1
            continue

        img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3,
                                    preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
        img_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(args.device)

        box_1024 = np.array([box], dtype=np.float32) / np.array([W, H, W, H]) * 1024
        embed = model.image_encoder(img_t)
        seg = medsam_infer(model, embed, box_1024, H, W)
        Image.fromarray(seg * 255).save(out_dir / f"{case}.png")
        n_box += 1

    print(f"Wrote {n_box + n_empty} preds -> {out_dir}  (prompted {n_box}, empty {n_empty})")


if __name__ == "__main__":
    main()
