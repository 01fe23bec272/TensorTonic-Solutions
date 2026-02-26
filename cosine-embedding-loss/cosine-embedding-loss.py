import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
      # Convert to numpy arrays
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # Compute cosine similarity
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    # Avoid division by zero
    if norm_x1 == 0 or norm_x2 == 0:
        return 0.0

    cos_sim = dot_product / (norm_x1 * norm_x2)

    # Compute loss based on label
    if label == 1:
        loss = 1 - cos_sim
    elif label == -1:
        loss = max(0.0, cos_sim - margin)
    else:
        raise ValueError("label must be +1 or -1")

    return float(loss)