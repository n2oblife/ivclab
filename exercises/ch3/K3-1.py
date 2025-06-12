import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from ivclab.utils import imread
from ivclab.utils.metrics import calc_psnr

def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(dct_coeff):
    return idct(idct(dct_coeff.T, norm='ortho').T, norm='ortho')

# Load grayscale image
img = imread('data/lena_gray.tif')
img = img.astype(np.float32)

# Apply 2D DCT
dct_coeff = dct2(img)
flattened = np.abs(dct_coeff).flatten()
sorted_indices = np.argsort(flattened)[::-1]  # Descending order

# Percentages of top coefficients to remove
percentages = [0.01, 0.05, 0.10]
reconstructions = []

for perc in percentages:
    dct_mod = dct_coeff.copy()
    N = int(perc * dct_coeff.size)
    top_indices = sorted_indices[:N]

    # Zero out top N coefficients
    flat_dct = dct_mod.flatten()
    flat_dct[top_indices] = 0
    dct_mod = flat_dct.reshape(dct_coeff.shape)

    # Inverse DCT
    recon = idct2(dct_mod)
    recon = np.clip(recon, 0, 255)
    reconstructions.append((perc, recon, calc_psnr(img, recon)))

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

for i, (p, recon, score) in enumerate(reconstructions):
    plt.subplot(2, 2, i+2)
    plt.imshow(recon, cmap='gray')
    plt.title(f"Removed Top {int(p*100)}%\nPSNR: {score:.2f} dB")
    plt.axis('off')

plt.tight_layout()
plt.show()
