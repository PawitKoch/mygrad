import numpy as np


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal mask for attention mechanisms.

    The mask is a lower-triangular matrix where positions that should not be attended to
    are set to -inf, and valid positions are set to 0. This ensures that each position
    can only attend to previous positions (including itself) in the sequence.

    Example
    -------
    For seq_len = 4, the mask will be:

        [[ 0., -inf, -inf, -inf],
         [ 0.,   0., -inf, -inf],
         [ 0.,   0.,   0., -inf],
         [ 0.,   0.,   0.,   0.]]
    """
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


def unbroadcast(grad, target_shape):
    """
    Reverse the effect of broadcasting by summing gradients along broadcast dimensions.

    When numpy broadcasts during forward pass, gradients need to be summed back
    to match the original shape. This is crucial for correct backpropagation!

    Example 1 - Adding leading dimensions:
        Forward:  a(5,) * b(2,3,5) -> broadcasts a: (5,) -> (1,1,5) -> (2,3,5)
        Backward: grad(2,3,5) -> sum axes 0,1 -> (5,) for 'a'

    Example 2 - Dimension size 1:
        Forward:  a(3,1) + b(3,4) -> broadcasts a's axis 1 from 1->4
        Backward: grad(3,4) -> sum axis 1 with keepdims -> (3,1) for 'a'

    Example 3 - Both:
        Forward:  a(1,5) * b(2,3,5) -> broadcasts a: (1,5) -> (1,1,5) -> (2,3,5)
        Backward: grad(2,3,5) -> sum axis 0 -> (3,5) -> sum axis 0 keepdims -> (1,5)

    Algorithm:
    1. Remove leading dimensions that were added (grad.ndim > target.ndim)
    2. Sum along dimensions where target=1 but grad>1 (dimension was broadcast)
    """
    g = grad

    # Step 1: Remove leading dimensions added by broadcasting
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)

    # Step 2: Sum dimensions that were size-1 but got broadcast to size>1
    for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
        if ts == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)

    return g
