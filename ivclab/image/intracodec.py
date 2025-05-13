import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.quantization import PatchQuant
from ivclab.utils import ZigZag, Patcher
from ivclab.signal import DiscreteCosineTransform
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb

class IntraCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8)
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape

        self.dct = DiscreteCosineTransform()
        self.quant = PatchQuant(quantization_scale=quantization_scale)
        self.zigzag = ZigZag()
        self.zerorun = ZeroRunCoder(end_of_block=end_of_block, block_size= block_shape[0] * block_shape[1])
        self.huffman = HuffmanCoder(lower_bound=bounds[0])
        self.patcher = Patcher()

    def image2symbols(self, img: np.array, is_source_rgb=True):
        """
        Computes the symbol representation of an image by applying rgb2ycbcr,
        DCT, Quantization, ZigZag and ZeroRunEncoding in order.

        img: np.array of shape [H, W, C]

        returns:
            symbols: List of integers
        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return symbols
    
    def symbols2image(self, symbols, original_shape):
        """
        Reconstructs the original image from the symbol representation
        by applying ZeroRunDecoding, Inverse ZigZag, Dequantization and 
        IDCT, ycbcr2rgb in order. The argument original_shape is required to compute 
        patch_shape, which is needed by ZeroRunDecoding to correctly 
        reshape the input image from blocks.

        symbols: List of integers
        original_shape: List of 3 elements that contains H, W and C
        
        returns:
            reconstructed_img: np.array of shape [H, W, C]
        """
        patch_shape = [original_shape[0] // 8, original_shape[1] // 8, original_shape[2]]
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return reconstructed_img
    
    def train_huffman_from_image(self, training_img, is_source_rgb=True):
        """
        Finds the symbols representing the image, extracts the 
        probability distribution of them and trains the huffman coder with it.

        training_img: np.array of shape [H, W, C]

        returns:
            Nothing
        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE

    def intra_encode(self, img: np.array, return_bpp = False, is_source_rgb=True):
        """
        Encodes an image to a bitstream and return it by converting it to
        symbols and compressing them with the Huffman coder.

        img: np.array of shape [H, W, C]

        returns:
            bitstream: List of integers produced by the Huffman coder
        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return bitstream
    
    def intra_decode(self, bitstream, original_shape):
        """
        Decodes an image from a bitstream by decoding it with the Huffman
        coder and reconstructing it from the symbols.

        bitstream: List of integers produced by the Huffman coder
        original_shape: List of 3 values that contain H, W, and C

        returns:
            reconstructed_img: np.array of shape [H, W, C]

        """
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return reconstructed_img
    
if __name__ == "__main__":
    from ivclab.utils import imread,calc_psnr
    import matplotlib.pyplot as plt

    lena = imread(f'data/lena.tif')
    lena_small = imread(f'data/lena_small.tif')
    intracodec = IntraCodec(quantization_scale=0.15)
    intracodec.train_huffman_from_image(lena_small)
    symbols, bitsize = intracodec.intra_encode(lena, return_bpp=True)
    print(len(symbols))
    reconstructed_img = intracodec.intra_decode(symbols, lena.shape)
    psnr = calc_psnr(lena, reconstructed_img)
    print(f"PSNR: {psnr:.4f} dB, bpp: {bitsize / (lena.size / 3)}")
    