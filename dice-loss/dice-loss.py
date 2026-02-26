import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Flatten arrays (important for multi-dimensional masks)
    p = p.flatten()
    y = y.flatten()

    # Compute intersection
    intersection = np.sum(p * y)

    # Compute Dice coefficient
    dice = (2.0 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    # Dice loss
    return float(1.0 - dice)