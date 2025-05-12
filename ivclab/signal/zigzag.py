import numpy as np

def zigzag_scan(block):
    """
    Perform zig-zag scan on an 8x8 block.
    
    Parameters:
        block (np.ndarray): 8x8 matrix of DCT coefficients.
    
    Returns:
        np.ndarray: 1D array of coefficients in zig-zag order.
    """
    assert block.shape == (8, 8), "Input must be an 8x8 block"
    
    zigzag_indices = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    
    return np.array([block[i, j] for i, j in zigzag_indices])

