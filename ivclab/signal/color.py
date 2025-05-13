import numpy as np

def rgb2gray(image: np.array):
    """
    Computes the grayscale version of the image. 

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, 1]
    """
    output_image = np.mean(image, axis=-1, keepdims=True)
    return output_image

def rgb2ycbcr(image: np.array):
    """
    Converts an RGB image to its YCbCr version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)

    # Define conversion matrix and offset for BT.601
    conversion_matrix = np.array([
        [  0.299,     0.587,     0.114   ],
        [ -0.168736, -0.331264,  0.5     ],
        [  0.5,     -0.418688, -0.081312 ]
    ])

    offset = np.array([0, 128, 128])

    # Apply the transformation
    output_image = image @ conversion_matrix.T + offset
    return output_image

def ycbcr2rgb(image: np.array):
    """
    Converts an YCbCr image to its RGB version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)

    # Separate Y, Cb, Cr channels
    Y = image[:, :, 0]
    Cb = image[:, :, 1] - 128.0
    Cr = image[:, :, 2] - 128.0

    # Apply inverse transform
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    # Stack and clip the result to valid RGB range
    output_image = np.stack([R, G, B], axis=-1)
    output_image = np.clip(output_image, 0, 255)
    return output_image