from ivclab.utils import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from ivclab.utils.metrics import calc_psnr

def rgb_to_ycbcr_ict(rgb):
    """
    Converts an RGB image to YCbCr color space using the Irreversible Color Transform (ICT).

    Parameters:
        rgb (np.ndarray): Input image in RGB format.

    Returns:
        Tuple of np.ndarrays: (Y, Cb, Cr) components as float32 arrays.
    """
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.16875 * R - 0.33126 * G + 0.5 * B
    Cr =  0.5 * R - 0.41869 * G - 0.08131 * B
    return Y, Cb, Cr

def ycbcr_to_rgb_ict(Y, Cb, Cr):
    """
    Converts YCbCr components back to RGB using the inverse ICT.

    Parameters:
        Y, Cb, Cr (np.ndarray): YCbCr components.

    Returns:
        np.ndarray: Reconstructed RGB image.
    """
    R = Y + 1.402 * Cr
    G = Y - 0.34413 * Cb - 0.71414 * Cr
    B = Y + 1.772 * Cb
    return np.stack((R, G, B), axis=2)

def mirror_pad(image, pad=4):
    """
    Pads an image symmetrically to avoid border effects in resampling.

    Parameters:
        image (np.ndarray): Input image or channel.
        pad (int): Padding width.

    Returns:
        np.ndarray: Padded image.
    """
    return np.pad(image, pad_width=pad, mode='symmetric')

def subsample_channel(channel, factor=2, pad=4):
    """
    Subsamples a single-channel image (e.g., Cb or Cr) with padding to avoid edge artifacts.

    Parameters:
        channel (np.ndarray): Input image channel to be subsampled.
        factor (int): Downsampling factor (default is 2).
        pad (int): Padding size for symmetric extension.

    Returns:
        np.ndarray: Subsampled channel, rounded to integers.
    """
    padded = mirror_pad(channel, pad)
    down_H = resample(padded, padded.shape[0] // factor, axis=0)
    downsampled = resample(down_H, padded.shape[1] // factor, axis=1)

    # Crop back to expected subsampled shape
    crop_y = (downsampled.shape[0] - channel.shape[0] // factor) // 2
    crop_x = (downsampled.shape[1] - channel.shape[1] // factor) // 2
    return np.round(downsampled[crop_y:-crop_y, crop_x:-crop_x])

def ict_transform(image: np.ndarray, show=False):
    """
    Performs ICT transformation and chroma subsampling on an image.

    Parameters:
        image (np.ndarray): Input RGB image.
        show (bool): If True, displays Y, Cb, Cr channels.

    Returns:
        Tuple: (Y, Cb_subsampled, Cr_subsampled)
    """
    Y, Cb, Cr = rgb_to_ycbcr_ict(image)
    Cb_sub = subsample_channel(Cb)
    Cr_sub = subsample_channel(Cr)

    # Round values for storage
    Y = np.round(Y)
    Cb_sub = np.round(Cb_sub)
    Cr_sub = np.round(Cr_sub)

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.title("Y"); plt.imshow(Y, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Cb (subsampled)"); plt.imshow(Cb_sub, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Cr (subsampled)"); plt.imshow(Cr_sub, cmap="gray"); plt.axis("off")
        plt.tight_layout()
        plt.show()

    return Y, Cb_sub, Cr_sub

def upsample_channel(channel, target_shape, pad=4):
    """
    Upsamples a channel to a target shape using symmetric padding and resampling.

    Parameters:
        channel (np.ndarray): Subsampled image channel.
        target_shape (tuple): Desired shape after upsampling (height, width).
        pad (int): Padding width.

    Returns:
        np.ndarray: Upsampled channel.
    """
    padded = mirror_pad(channel, pad)
    up_H = resample(padded, target_shape[0] + 2 * pad, axis=0)
    upsampled = resample(up_H, target_shape[1] + 2 * pad, axis=1)
    return upsampled[pad:-pad, pad:-pad]

def reconstruction_pipe(Y, Cb_sub, Cr_sub):
    """
    Reconstructs an RGB image from Y, Cb, Cr components.

    Parameters:
        Y (np.ndarray): Luminance component.
        Cb_sub, Cr_sub (np.ndarray): Subsampled chrominance components.

    Returns:
        np.ndarray: Reconstructed RGB image.
    """
    Cb_up = upsample_channel(Cb_sub, Y.shape)
    Cr_up = upsample_channel(Cr_sub, Y.shape)
    reconstructed = ycbcr_to_rgb_ict(Y, Cb_up, Cr_up)
    return np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

def plot_diff_ict(image, reconstructed):
    """
    Displays the original and reconstructed images side by side and prints PSNR.

    Parameters:
        image (np.ndarray): Original image.
        reconstructed (np.ndarray): Reconstructed image.
    """
    original = image.astype(np.float32)
    recon_float = reconstructed.astype(np.float32)
    image_psnr = calc_psnr(original, recon_float, maxval=255)

    print(f"PSNR (Reconstructed vs Original): {image_psnr:.2f} dB")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.title("Original"); plt.imshow(image); plt.axis("off")
    plt.subplot(1, 2, 2); plt.title(f"Reconstructed; PSNR={image_psnr:.2f} dB"); plt.imshow(reconstructed); plt.axis("off")
    plt.tight_layout()
    plt.show()

def codec_ict(image, show=False):
    Y, Cb_sub, Cr_sub = ict_transform(image, show)
    reconstructed = reconstruction_pipe(Y, Cb_sub, Cr_sub)
    return reconstructed

if __name__ == "__main__":
    image = imread("data/sail.tif")

    Y, Cb_sub, Cr_sub = ict_transform(image, show=True)
    reconstructed = reconstruction_pipe(Y, Cb_sub, Cr_sub)
    plot_diff_ict(image, reconstructed)
