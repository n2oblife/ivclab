import numpy as np
from scipy.signal import decimate
from ivclab.signal import rgb2ycbcr, ycbcr2rgb

def single_pixel_predictor(image):
    """
    Creates a residual image after a single pixel predictor for overlapping 
    pixel pairs. The right pixel is predicted from the left pixel with the formula
    R_pred = L * a1 where a1=1. This function returns the residual R - R_pred. For
    the first pixels of each row who don't have a left neighbor, it copies the values
    from the original image instead of making a prediction

    image: np.array of shape [H, W, C]

    returns 
        residual_image: np.array of shape [H, W, C]
    """
    # Convert image to floating points
    image = image * 1.0

    a1 = 1.0

    # Create residual image
    residual_image = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    channel = image.shape[2] if image.ndim == 3 else 1
    for c in range(channel):
        img_channel = image[:, :, c] if channel > 1 else image
        prediction = img_channel[:, :-1] * a1
        image_channel = img_channel[:, 1:]
        residual_image = image_channel - prediction
    # YOUR CODE ENDS HERE

    residual_image = np.round(np.clip(residual_image, -255, 255))

    return residual_image

def _predict_from_neighbors(original, coefficients):
    """
    Helper function for the three pixel predictor. Here is the main computation:

    prediction(current) = coefficients * reconstruction(previous)
    error(current) = round(original(current) - prediction(current))
    reconstruction(current) = prediction(current) + error(current)

    We need to create two arrays, reconstruction and residual_error. They are already
    initialized such that the top row and the leftmost column of the original image
    is copied to them.
    
    It applies this over all pixels from top-left to bottom-right in order.

    Hint: Start from the second index in "for loops" of both directions

    original: np.array of shape [H, W, C]
    reconstruction: np.array of shape [H, W, C]
    residual_error: np.array of shape [H, W, C]
    coefficients: list of 3 floating point numbers (see lab slides for what they represent)

    returns 
        residual_error: np.array of shape [H, W, C]
    """
    H, W, C = original.shape

    reconstruction = np.zeros_like(original)
    reconstruction[0,:,:] = original[0,:,:]
    reconstruction[:,0,:] = original[:,0,:]

    residual_error = np.copy(reconstruction)

    # YOUR CODE STARTS HERE
    for i in range(1, H):
        for j in range(1, W):
            for c in range(C):
                top = reconstruction[i - 1, j, c]
                left = reconstruction[i, j - 1, c]
                top_left = reconstruction[i - 1, j - 1, c]
                prediction = (coefficients[0] * left +
                                coefficients[1] * top +
                                coefficients[2] * top_left)
                error = np.round(original[i, j, c] - prediction)
                reconstruction[i, j, c] = prediction + error
                residual_error[i, j, c] = error
    # YOUR CODE STARTS HERE
    return residual_error

def three_pixels_predictor(image, subsample_color_channels=False):
    """
    Creates a residual image after a three pixels predictor.

    1. Convert the input image to YCbCr color space
    2. If subsample_color_channels, then subsample the Cb and Cr channels
        by 2, similar to the yuv420codec (use scipy.signal.decimate)
    3. Apply three pixel prediction with the given coefficients for Y and CbCr channels.
        You must use _predict_from_neighbors helper function
    4. Return the residual error images

    image: np.array of shape [H, W, C]

    returns 
        residual_image_Y: np.array of shape [H, W, 1]
        residual_image_CbCr: np.array of shape [H, W, 2] (or [H // 2, W // 2, 2] if subsampled)
    """
    # Convert image to floating points
    image = image * 1.0

    coefficients_Y = [7/8, -4/8, 5/8]
    coefficients_CbCr = [3/8, -2/8, 7/8]

    # YOUR CODE STARTS HERE

    ycbcr_image = rgb2ycbcr(image)
    Y = ycbcr_image[:, :, 0:1]
    CbCr = ycbcr_image[:, :, 1:3]

    # Predict Y channel
    residual_image_Y = _predict_from_neighbors(Y, coefficients_Y)

    # Subsample and predict CbCr
    if subsample_color_channels:
        Cb = decimate(decimate(CbCr[:, :, 0], 2, axis=0), 2, axis=1)
        Cr = decimate(decimate(CbCr[:, :, 1], 2, axis=0), 2, axis=1)
        CbCr_sub = np.stack([Cb, Cr], axis=-1)
        residual_image_CbCr = _predict_from_neighbors(CbCr_sub, coefficients_CbCr)
    else:
        residual_image_CbCr = _predict_from_neighbors(CbCr, coefficients_CbCr)

    # YOUR CODE ENDS HERE

    residual_image_Y = np.round(np.clip(residual_image_Y, -255, 255)).astype(np.int32)
    residual_image_CbCr = np.round(np.clip(residual_image_CbCr, -255, 255)).astype(np.int32)

    return residual_image_Y, residual_image_CbCr