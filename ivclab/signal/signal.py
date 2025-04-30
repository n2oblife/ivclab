import numpy as np
from scipy.signal import convolve2d

def downsample(image):
    """
    Selects only the even positioned pixels of an image
    to downsample it.

    image: np.array of shape [H, W, C]

    returns 
        downsampled_image: np.array of shape [H // 2, W // 2, C]
    """
    if image.ndim == 3:
        return image[0::2, 0::2, :]
    if image.ndim == 2:
        return image[0::2, 0::2]

def upsample(image):
    """
    Upsamples an image by filling new pixels with zeroes.

    image: np.array of shape [H, W, C]

    returns 
        upsampled_image: np.array of shape [H * 2, W * 2, C]
    """
    if image.ndim == 3:
        H, W, C = image.shape
        upsampled_image = np.zeros((2 * H, 2 * W, C), dtype=image.dtype)
        upsampled_image[0::2, 0::2, :] = image
    elif image.ndim == 2:
        H, W = image.shape
        upsampled_image = np.zeros((2 * H, 2 * W), dtype=image.dtype)
        upsampled_image[0::2, 0::2] = image
    return upsampled_image

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