import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(base) / d_model)
    )

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(position * div_term)

    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(position * div_term)
    else:
        pe[:, 1::2] = np.cos(position * div_term[:-1])

    return pe