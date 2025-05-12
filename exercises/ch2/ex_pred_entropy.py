import numpy as np
from ivclab.utils import imread
from ivclab.entropy import calc_entropy
import matplotlib.pyplot as plt

def prediction_error(image):
    """Compute prediction error using S1 as predictor for S0"""
    errors = []
    for c in range(3):  # R, G, B channels
        channel = image[:, :, c]
        pred = channel[:, :-1]
        actual = channel[:, 1:]
        error = actual.astype(int) - pred.astype(int)  # Error range: -255 to 255
        errors.append(error.flatten())
    return np.concatenate(errors)

def compute_error_pmf(errors, bins=np.arange(-255, 257)):
    """Compute normalized histogram (PMF) for prediction errors"""
    hist, _ = np.histogram(errors, bins=bins)
    pmf = hist / np.sum(hist)
    return pmf

if __name__ == "__main__":
    # Load Lena image (RGB)
    image = imread("data/lena.tif")  # adjust path as needed

    # Step 1: Compute prediction error using S1
    errors = prediction_error(image)

    # Step 2: Compute PMF of errors
    pmf = compute_error_pmf(errors)

    # Step 3: Compute entropy of error (minimum average code length)
    entropy = calc_entropy(pmf)

    print(f"Entropy of prediction error using left neighbor : {entropy:.2f} bits/pixel")