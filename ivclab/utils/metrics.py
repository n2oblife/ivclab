import numpy as np

def calc_mse(orig: np.array, rec: np.array):
    """
    Computes the Mean Squared Error by taking the square of
    the difference between orig and rec, and averaging it
    over all the pixels.

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        mse: a scalar value
    """
    # YOUR CODE STARTS HERE
    # Convert grayscale to 3-channel if the other is RGB
    if orig.ndim == 2 and rec.ndim == 3:
        orig = np.stack([orig]*3, axis=-1)
    elif orig.ndim == 3 and rec.ndim == 2:
        rec = np.stack([rec]*3, axis=-1)
    
    assert orig.shape == rec.shape, f"Image shapes don't match after processing: {orig.shape} vs {rec.shape}"
    mse = np.mean((orig.astype(np.float64) - rec.astype(np.float64)) ** 2)   
    # YOUR CODE ENDS HERE
    return mse

def calc_psnr(orig: np.array, rec:np.array, maxval=255):
    """
    Computes the Peak Signal Noise Ratio by computing
    the MSE and using it in the formula from the lectures.

    > **_ Warning _**: Assumes the signals are in the 
    range [0, 255] by default

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        psnr: a scalar value
    """
    # YOUR CODE STARTS HERE
    psnr = 20*np.log10(maxval / np.sqrt(calc_mse(orig, rec)))
    # YOUR CODE ENDS HERE
    return psnr