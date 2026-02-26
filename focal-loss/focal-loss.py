import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    # Convert to numpy arrays
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Avoid log(0)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Compute p_t
    p_t = np.where(y == 1, p, 1 - p)

    # Compute focal loss
    loss = -((1 - p_t) ** gamma) * np.log(p_t)

    # Return mean loss
    return float(np.mean(loss))