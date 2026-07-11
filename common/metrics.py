"""Segmentation metrics shared across models.

Dice is computed over binary masks. For the empty-GT / empty-pred case we
return 1.0 (both agree there is nothing), which matches the SIIM-ACR
competition convention; a nonempty prediction against empty GT scores 0.
"""
import numpy as np


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)
