import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    
    # Convert to numpy arrays
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Avoid division by zero or log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return float(kl)