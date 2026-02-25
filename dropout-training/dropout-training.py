import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)

    if rng is None:
        rand_vals = np.random.rand(*x.shape)
    else:
        rand_vals = rng.random(x.shape)

    keep_prob = 1 - p

    mask = np.where(rand_vals < keep_prob, 1.0 / keep_prob, 0.0)

    output = x * mask

    return output, mask
        