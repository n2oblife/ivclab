import matplotlib.pyplot as plt
import numpy as np
from ivclab.utils import imread
from ivclab.signal import rgb2gray, downsample, upsample
from scipy.signal import convolve2d
from scipy.ndimage import zoom


def plot_frequency_response(kernel: np.ndarray):
    """
    Plots the frequency response of a filter kernel.
    
    Parameters:
        kernel (np.ndarray): 2D filter kernel
    """
    # Zero-pad kernel to higher resolution for visualization
    size = 256
    padded = np.zeros((size, size))
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel

    # Compute FFT and shift zero frequency to center
    freq_response = np.abs(np.fft.fftshift(np.fft.fft2(padded)))

    # Plot
    plt.figure(figsize=(6, 5))
    plt.title("Frequency Response of Low-pass Filter")
    plt.matshow(freq_response, cmap='viridis', fignum=0)
    plt.colorbar()
    plt.show()

def conv_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Applies a filter to a 2D image or each channel of an RGB image."""
    kernel /= np.sum(kernel)  # Normalize kernel
    
    if image.ndim == 3:  # RGB image
        # Apply convolution to each color channel independently
        return np.stack([convolve2d(image[:, :, c], kernel, mode='same', boundary='symm') for c in range(image.shape[2])], axis=2)
    else:  # Grayscale image
        return convolve2d(image, kernel, mode='same', boundary='symm')

def plot_differences(img: np.ndarray, filtered: np.ndarray, show=False):
    """Displays the original, filtered, and the difference between them."""
    difference = img - filtered
    
    # Normalize images to be in the [0, 1] range if they are float images
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 1)
        filtered = np.clip(filtered, 0, 1)
        difference = np.clip(difference, -1, 1)  # Difference can be negative
    elif img.dtype == np.uint8:
        # For uint8 images, normalize to [0, 255]
        img = np.clip(img, 0, 255)
        filtered = np.clip(filtered, 0, 255)
        difference = np.clip(difference, -255, 255)  # Difference can also go negative
    
    if show:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Filtered")
        plt.imshow(filtered, cmap='gray' if filtered.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Difference (Original - Filtered)")
        plt.imshow(difference + 0.5, cmap='gray' if difference.ndim == 2 else None)  # Shift for visibility
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def interpolation_upsample(image: np.ndarray) -> np.ndarray:
    """
    Upsamples an image by a factor of 2 using bilinear interpolation.
    
    Parameters:
        image: np.ndarray of shape [H, W] or [H, W, C]
    
    Returns:
        upsampled_image: np.ndarray of shape [2*H, 2*W] or [2*H, 2*W, C]
    """
    if image.ndim == 3:  # For color images (RGB)
        return np.stack([
            zoom(image[:, :, c], 2, order=1)  # order=1 -> bilinear interpolation
            for c in range(image.shape[2])
        ], axis=2)
    else:  # Grayscale
        return zoom(image, 2, order=1)

def codec(img: np.ndarray, kernel: np.ndarray, show=False):
    """Applies filtering, downsampling, and upsampling to the image."""
    filtered = conv_filter(img, kernel)
    down_img = downsample(filtered)
    up_img = interpolation_upsample(down_img)
    
    # Normalize images to the correct range for visualization:
    if img.dtype == np.float32 or img.dtype == np.float64:
        # If the image is in floating point, scale to [0, 1]
        img = np.clip(img, 0, 1)
        filtered = np.clip(filtered, 0, 1)
        down_img = np.clip(down_img, 0, 1)
        up_img = np.clip(up_img, 0, 1)
    elif img.dtype == np.uint8:
        # If image is uint8, make sure values are in [0, 255]
        img = np.clip(img, 0, 255)
        filtered = np.clip(filtered, 0, 255)
        down_img = np.clip(down_img, 0, 255)
        up_img = np.clip(up_img, 0, 255)
    else:
        # If data type is unexpected, force a normalization (just in case)
        img = np.clip(img, 0, 255)
        filtered = np.clip(filtered, 0, 255)
        down_img = np.clip(down_img, 0, 255)
        up_img = np.clip(up_img, 0, 255)
    
    if show:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Filtered")
        plt.imshow(filtered, cmap='gray' if filtered.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Downsampled")
        plt.imshow(down_img, cmap='gray' if down_img.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Upsampled")
        plt.imshow(up_img, cmap='gray' if up_img.ndim == 2 else None)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return filtered, down_img, up_img


if __name__ == "__main__":
    img = imread("data/satpic1.bmp")
    
    # Convert to grayscale if the image is RGB
    gray_img = rgb2gray(img) if img.ndim == 3 else img

    # Low-pass filter kernel
    low_pass_kernel = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=float
    )
    # Apply the codec function with the image and kernel
    filtered, down_img, up_img = codec(img, low_pass_kernel, show=True)
    plot_differences(img, filtered, True)