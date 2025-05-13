import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import zoom

def downsample(image, factor=2):
    """
    Selects only the even positioned pixels of an image
    to downsample it.

    image: np.array of shape [H, W, C]
    factor: int sampling factor

    returns 
        downsampled_image: np.array of shape [H // 2, W // 2, C]
    """
    if image.ndim == 3:
        return image[0::factor, 0::factor, :]
    if image.ndim == 2:
        return image[0::factor, 0::factor]

def upsample(image, factor=2):
    """
    Upsamples an image by filling new pixels with zeroes.

    image: np.array of shape [H, W, C]

    returns 
        upsampled_image: np.array of shape [H * factor, W * factor, C]
    """
    if image.ndim == 3:
        H, W, C = image.shape
        upsampled_image = np.zeros((factor * H, factor * W, C), dtype=image.dtype)
        upsampled_image[0::factor, 0::factor, :] = image
    elif image.ndim == 2:
        H, W = image.shape
        upsampled_image = np.zeros((factor * H, factor * W), dtype=image.dtype)
        upsampled_image[0::factor, 0::factor] = image
    return upsampled_image

def interpolation_upsample(image: np.ndarray, factor=2, classic=False) -> np.ndarray:
    """
    Upsamples an image by a factor using bilinear interpolation.
    
    Parameters:
        image: np.ndarray of shape [H, W] or [H, W, C]
    
    Returns:
        upsampled_image: np.ndarray of shape [2*H, 2*W] or [2*H, 2*W, C]
    """
    if image.ndim == 3:  # For color images (RGB)
        return np.stack([
            zoom(image[:, :, c], factor, order=1)  # order=1 -> bilinear interpolation
            for c in range(image.shape[2])
        ], axis=2) if not classic else upsample(image)
    else:  # Grayscale
        return zoom(image, factor, order=1)

def lowpass_filter(image, kernel):
    """
    Applies a given kernel on each channel of an image separately
    with convolution operation. 

    image: np.array of shape [H, W, C]
    kernel: np.array of shape [kernel_height, kernel_width]

    returns 
        filtered: np.array of shape [H, W, C]
    """
    filtered = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    kernel /= np.sum(kernel) # Normalize the kernel
    filtered = convolve2d(image, kernel, mode='same', boundary='symm')    
    # YOUR CODE ENDS HERE
    return filtered

class FilterPipeline:

    def __init__(self, kernel):
        """
        Initializes a filtering pipeline with the given kernel 

        kernel: np.array of shape [kernel_height, kernel_width]
        """
        self.kernel = kernel / np.sum(kernel)

    def filter_img(self, image: np.ndarray, prefilter: bool=True):
        """
        Applies prefiltering to an image (optional), downsamples, 
        upsamples and filters the output with a lowpass filter. 

        image: np.array of shape [H, W, C]

        returns 
            output_image: np.array of shape [H, W, C]
        """
        # Cast image to floating point
        output = image * 1.0

        # YOUR CODE STARTS HERE


        #YOUR CODE ENDS HERE

        # Cast output to integer again
        output = np.round(output).astype(np.uint8)
        return output