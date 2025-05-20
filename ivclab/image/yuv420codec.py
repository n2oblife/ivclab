import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample
from ivclab.utils import imread
from ivclab.utils.metrics import calc_psnr
from ivclab.signal import rgb2gray, downsample, upsample

def rgb2ycbcr(image:np.ndarray):
    """
    Convert RGB to YCbCr using ITU-R BT.601 conversion
    """
    mat = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])
    ycbcr = image @ mat.T
    ycbcr[:, :, [1, 2]] += 128.0
    return ycbcr

def ycbcr2rgb(image:np.ndarray):
    """
    Convert YCbCr to RGB using ITU-R BT.601 inverse conversion
    """
    image = image.copy()
    image[:, :, [1, 2]] -= 128.0
    mat_inv = np.array([[1.0, 0.0, 1.402],
                        [1.0, -0.344136, -0.714136],
                        [1.0, 1.772, 0.0]])
    rgb = image @ mat_inv.T
    return np.clip(rgb, 0, 255)

def pad_image(img, pad=4):
    return np.pad(img, ((pad, pad), (pad, pad)), mode='symmetric')

def crop_image(img, pad=4):
    return img[pad:-pad, pad:-pad]

def yuv420compression(image: np.ndarray):
    """
    Steps:
    1. Convert an image from RGB to YCbCr
    2. Compress the image
        A. Pad the image with 4 pixels symmetric pixels on each side
        B. Downsample only Cb and Cr channels with prefiltering (use scipy.signal.decimate for it)
        C. Crop the image 2 pixels from each side to get rid of padding
    3. Apply rounding to Y, Cb and Cr channels
    4. Decompress the image
        A. Pad the image with 4 pixels symmetric pixels on each side
        B. Upsample Cb and Cr channels (use scipy.signal.resample for it)
        C. Crop the image 2 pixels from each side to get rid of padding
    5. Convert the YCbCr image back to RGB

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, C]
    """
    # Cast image to floating point
    image = image * 1.0

    # Step 1: Convert RGB to YCbCr
    ycbcr = rgb2ycbcr(image)
    Y = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    # Step 2A: Symmetric padding
    pad = 4
    Cb_padded = pad_image(Cb, pad)
    Cr_padded = pad_image(Cr, pad)

    # Step 2B: Downsample using decimate (factor 2 in both dimensions)
    Cb_ds = decimate(decimate(Cb_padded, 2, axis=0, ftype='fir', zero_phase=True), 2, axis=1, ftype='fir', zero_phase=True)
    Cr_ds = decimate(decimate(Cr_padded, 2, axis=0, ftype='fir', zero_phase=True), 2, axis=1, ftype='fir', zero_phase=True)

    # Step 2C: Crop back Y (no padding for Y)
    Y = crop_image(pad_image(Y, pad), pad)

    # Step 3: Round and store
    Y = np.round(Y)
    Cb_ds = np.round(Cb_ds)
    Cr_ds = np.round(Cr_ds)

    # Step 4A: Symmetric padding before upsampling
    Cb_ds_pad = pad_image(Cb_ds, pad)
    Cr_ds_pad = pad_image(Cr_ds, pad)

    # Step 4B: Upsample back to original shape using resample
    Cb_us = resample(resample(Cb_ds_pad, Cb_padded.shape[0], axis=0), Cb_padded.shape[1], axis=1)
    Cr_us = resample(resample(Cr_ds_pad, Cr_padded.shape[0], axis=0), Cr_padded.shape[1], axis=1)

    # Step 4C: Crop padded upsampled result to match original size
    Cb_final = crop_image(Cb_us, pad)
    Cr_final = crop_image(Cr_us, pad)

    # Step 5: Combine and convert back to RGB
    ycbcr_out = np.stack([Y, Cb_final, Cr_final], axis=2)
    output = ycbcr2rgb(ycbcr_out)
    
    # Cast output to integer again
    output = np.round(output).astype(np.uint8)
    return output

if __name__ == "__main__":
    # Load images
    img1 = imread('data/sail.tif').astype(np.float64)
    img2 = imread('data/lena.tif').astype(np.float64)

    # Apply YUV 4:2:0 compression and reconstruction
    reconstructed1 = yuv420compression(img1)
    reconstructed2 = yuv420compression(img2)

    # Compute PSNR
    psnr1 = calc_psnr(img1, reconstructed1, maxval=255, channel=3)
    psnr2 = calc_psnr(img2, reconstructed2, maxval=255, channel=3)

    # Show results
    print(f"PSNR for 'sail.tif': {psnr1:.2f} dB")
    print(f"PSNR for 'lena.tif': {psnr2:.2f} dB")

    # Optional: visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(img1.astype(np.uint8))
    axes[0, 0].set_title("Original sail.tif")
    axes[0, 1].imshow(reconstructed1)
    axes[0, 1].set_title(f"Reconstructed sail.tif\nPSNR: {psnr1:.2f} dB")
    axes[1, 0].imshow(img2.astype(np.uint8))
    axes[1, 0].set_title("Original lena.tif")
    axes[1, 1].imshow(reconstructed2)
    axes[1, 1].set_title(f"Reconstructed lena.tif\nPSNR: {psnr2:.2f} dB")
    for ax in axes.ravel(): ax.axis('off')
    plt.tight_layout()
    plt.show()