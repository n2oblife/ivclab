import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ivclab.utils import imread
from ivclab.signal import rgb2gray, downsample, upsample

def show_fft(image, title='FFT Magnitude'):
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))  # log scale for visibility
    plt.imshow(magnitude, cmap='gray')
    plt.title(title)
    plt.axis('off')

if __name__ == "__main__":
    # --- Load and prepare the image ---
    img = imread("data/satpic1.bmp")
    gray = rgb2gray(img) if img.ndim == 3 else img
    downsample_factor = 4

    # --- (a) Downsample without filtering ---
    down_no_filter = downsample(gray, factor=downsample_factor)

    # --- (b) Downsample with Gaussian filtering ---
    filtered = gaussian_filter(gray, sigma=2.0)  # sigma chosen empirically
    down_with_filter = downsample(filtered, factor=downsample_factor)

    # --- Plot FFTs ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    show_fft(gray, 'Original FFT')

    plt.subplot(1, 3, 2)
    show_fft(down_no_filter, 'Aliased FFT (no filtering)')

    plt.subplot(1, 3, 3)
    show_fft(down_with_filter, 'Filtered FFT (antialiasing)')

    plt.tight_layout()
    plt.show()

    plt.imshow(down_no_filter, cmap='gray')
    plt.title("Aliased (No Filtering)")
