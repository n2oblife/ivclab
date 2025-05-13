import numpy as np

class PatchQuant:
    """
    An object that handles forward and inverse quantization 
    of a patched image where each pixel of the patch is quantized
    with different values depending on the given matrices
    """

    def __init__(self, quantization_scale = 1.0, luminance = None, chrominance = None):
        self.quantization_scale = quantization_scale
        self.luminance = luminance
        self.chrominance = chrominance

        if self.luminance is None:
            self.luminance = np.asarray([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 55, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]).astype(np.float32)

        if self.chrominance is None:
            self.chrominance = np.asarray([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 13, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ]).astype(np.float32)

    def get_quantization_table(self):
        quantization_table = np.stack([self.luminance, self.chrominance, self.chrominance], axis=0)
        quantization_table = quantization_table * self.quantization_scale
        return quantization_table

    def quantize(self, patched_img):
        """
        Takes a patchified image and applies quantization on the 
        luminance and chrominance channels. Make sure to call get_quantization_table to 
        compute the correctly scaled table before dividing input values with them. 
        The returned values must contain rounded integers.

        patched_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]

        returns:
            quantized_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]
        """
        quant_table = self.get_quantization_table()  # [3, 8, 8]
        
        # Broadcasting division for quantization
        quantized_img = np.round(patched_img / quant_table[None, None, :, :, :])
        return quantized_img.astype(np.int32)
    
    def dequantize(self, quantized_img):
        """
        Takes a patchified and quantized image and applies dequantization on the 
        luminance and chrominance channels. Make sure to call get_quantization_table to 
        compute the correctly scaled table before multiplying input values with them. 
        The returned values must contain rounded integers.

        quantized_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]

        returns:
            patched_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]
        """
        quant_table = self.get_quantization_table()  # [3, 8, 8]

        # Broadcasting multiplication for dequantization
        patched_img = quantized_img * quant_table[None, None, :, :, :]
        return patched_img.astype(np.int32)