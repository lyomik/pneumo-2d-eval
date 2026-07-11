"""SIIM-ACR Pneumothorax RLE mask encode/decode.

Matches the semantics of the competition's own mask_functions.py:
"start length start length ...", where each `start` is a RELATIVE offset
from the end of the previous run (not an absolute index), traversed in
column-major order. "-1" means no pneumothorax.

Decoded masks are returned in standard (height, width) image orientation.
"""
import numpy as np


def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
    """Decode one RLE string into a (height, width) binary mask (0/1), uint8."""
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle.strip() == "-1":
        return mask.reshape(width, height).T

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        mask[current_position:current_position + length] = 1
        current_position += length

    return mask.reshape(width, height).T


def mask2rle(mask: np.ndarray) -> str:
    """Encode a (height, width) binary mask (nonzero = foreground) into an RLE string.

    Inverse of rle2mask: emits relative offsets in column-major order.
    """
    pixels = mask.T.flatten()
    pixels = np.asarray(pixels != 0, dtype=np.int8)
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0]
    run_starts = changes[0::2]
    run_ends = changes[1::2]

    rle = []
    prev_end = 0
    for start, end in zip(run_starts, run_ends):
        rle.append(str(start - prev_end))
        rle.append(str(end - start))
        prev_end = end

    return " ".join(rle) if rle else "-1"
