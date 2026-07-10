"""SIIM-ACR Pneumothorax RLE mask encode/decode.

RLE format matches the competition's train-rle.csv: space-separated
"start length start length ..." pairs, column-major (Fortran) order,
1-indexed start positions. A value of "-1" means no pneumothorax.
"""
import numpy as np


def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
    """Decode one RLE string into a (height, width) binary mask (0/1), uint8."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle.strip() == "-1":
        return mask.reshape(height, width)

    tokens = [int(t) for t in rle.strip().split()]
    starts = tokens[0::2]
    lengths = tokens[1::2]
    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1

    return mask.reshape(width, height).T


def mask2rle(mask: np.ndarray) -> str:
    """Encode a (height, width) binary mask (0/1) into an RLE string."""
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)
