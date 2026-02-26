import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure y_pred is 2D (important for single sample case like [2])
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    # Number of samples
    n = len(y_true)

    # Avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Extract correct class probabilities
    correct_class_probs = y_pred[np.arange(n), y_true]

    # Compute cross-entropy
    loss = -np.mean(np.log(correct_class_probs))

    return loss
