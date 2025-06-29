import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.quantization import PatchQuant
from ivclab.utils import ZigZag, Patcher
from ivclab.signal import DiscreteCosineTransform
from ivclab.entropy import HuffmanCoder, stats_marg, smooth_pmf
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from einops import rearrange
import pickle

class IntraCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8)
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = None
        self.end_of_block = end_of_block
        self.block_shape = block_shape

        self.dct = DiscreteCosineTransform()
        self.quant = PatchQuant(quantization_scale=quantization_scale)
        self.zigzag = ZigZag()
        self.zerorun = ZeroRunCoder(end_of_block=end_of_block, block_size= block_shape[0] * block_shape[1])
        self.huffman = None
        self.patcher = Patcher()

    def image2symbols(self, img: np.array, is_source_rgb=True):
        """
        Computes the symbol representation of an image by applying rgb2ycbcr,
        DCT, Quantization, ZigZag and ZeroRunEncoding in order.

        img: np.array of shape [H, W, C] or [H, W] for grayscale

        returns:
            symbols: List of integers
        """
        print(f"🔍 DEBUG: image2symbols - Input shape: {img.shape}, dtype: {img.dtype}")

        # Convert RGB to YCbCr if needed
        img_ycbcr = rgb2ycbcr(img) if is_source_rgb else img
        print(f"🔍 DEBUG: After RGB->YCbCr - Shape: {img_ycbcr.shape}, dtype: {img_ycbcr.dtype}")

        # Ensure shape is [H, W, C], even for grayscale
        if img_ycbcr.ndim == 2:
            img_ycbcr = img_ycbcr[:, :, np.newaxis]
            print(f"🔍 DEBUG: Added channel dimension - Shape: {img_ycbcr.shape}")

        # Check if image dimensions are multiples of block size
        H, W, C = img_ycbcr.shape
        if H % self.block_shape[0] != 0 or W % self.block_shape[1] != 0:
            print(f"⚠️  WARNING: Image dimensions ({H}, {W}) are not multiples of block size {self.block_shape}")
            print(f"   This may cause block misalignment artifacts!")
            # Pad to make it a multiple of block size
            pad_h = (self.block_shape[0] - H % self.block_shape[0]) % self.block_shape[0]
            pad_w = (self.block_shape[1] - W % self.block_shape[1]) % self.block_shape[1]
            if pad_h > 0 or pad_w > 0:
                img_ycbcr = np.pad(img_ycbcr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                print(f"🔍 DEBUG: Padded image to shape: {img_ycbcr.shape}")

        print(f"🔍 DEBUG: Before patching - Shape: {img_ycbcr.shape}")
        patches = rearrange(img_ycbcr, '(h ph) (w pw) c -> h w c ph pw', ph=self.block_shape[0], pw=self.block_shape[1])
        print(f"🔍 DEBUG: After patching - Shape: {patches.shape}")

        dct_patches = self.dct.transform(patches)
        print(f"🔍 DEBUG: After DCT - Shape: {dct_patches.shape}")
        
        quantized = self.quant.quantize(dct_patches)
        print(f"🔍 DEBUG: After quantization - Shape: {quantized.shape}")
        
        zz_scanned = self.zigzag.flatten(quantized)
        print(f"🔍 DEBUG: After zigzag flatten - Shape: {zz_scanned.shape}")
        
        symbols = self.zerorun.encode(zz_scanned)
        print(f"🔍 DEBUG: After zero-run encode - Symbols: {len(symbols)}")

        return symbols

    
    def symbols2image(self, symbols, original_shape):
        """
        Reconstructs the original image from the symbol representation
        by applying ZeroRunDecoding, Inverse ZigZag, Dequantization and 
        IDCT, and optionally ycbcr2rgb if the input was RGB.

        symbols: List of integers
        original_shape: Tuple of 2 or 3 elements containing H, W, (C)

        returns:
            reconstructed_img: np.array of shape [H, W, C] or [H, W] if grayscale
        """
        print(f"🔍 DEBUG: symbols2image - Original shape: {original_shape}, Symbols: {len(symbols)}")
        
        # Determine image shape and color mode
        if len(original_shape) == 2:
            H, W = original_shape
            C = 1
            is_rgb = False
        else:
            H, W, C = original_shape
            is_rgb = True

        print(f"🔍 DEBUG: Parsed shape - H: {H}, W: {W}, C: {C}, is_rgb: {is_rgb}")

        patch_shape = [H // 8, W // 8, C]
        print(f"🔍 DEBUG: Patch shape: {patch_shape}")
        
        decoded = self.zerorun.decode(symbols, original_shape=patch_shape)
        print(f"🔍 DEBUG: After zero-run decode - Shape: {decoded.shape}")
        
        inv_zz = self.zigzag.unflatten(decoded)
        print(f"🔍 DEBUG: After zigzag unflatten - Shape: {inv_zz.shape}")
        
        dequant = self.quant.dequantize(inv_zz)
        print(f"🔍 DEBUG: After dequantization - Shape: {dequant.shape}")
        
        ycbcr = self.dct.inverse_transform(dequant)
        print(f"🔍 DEBUG: After IDCT - Shape: {ycbcr.shape}")
        
        ycbcr = rearrange(ycbcr, 'hp wp c h w -> (hp h) (wp w) c')
        print(f"🔍 DEBUG: After rearrange - Shape: {ycbcr.shape}")

        # Crop back to original size if we padded
        if ycbcr.shape[0] != H or ycbcr.shape[1] != W:
            print(f"🔍 DEBUG: Cropping from {ycbcr.shape} to ({H}, {W}, {C})")
            ycbcr = ycbcr[:H, :W, :]

        if C == 1:
            # If it's [H, W, 1], squeeze; otherwise, leave as-is
            if ycbcr.ndim == 3 and ycbcr.shape[2] == 1:
                reconstructed_img = ycbcr[:, :, 0]
                print(f"🔍 DEBUG: Squeezed single channel - Shape: {reconstructed_img.shape}")
            else:
                reconstructed_img = ycbcr
        elif is_rgb:
            reconstructed_img = ycbcr2rgb(ycbcr)
            print(f"🔍 DEBUG: After YCbCr->RGB - Shape: {reconstructed_img.shape}")
        else:
            reconstructed_img = ycbcr  # fallback, possibly YCbCr format

        print(f"🔍 DEBUG: Final output - Shape: {reconstructed_img.shape}, dtype: {reconstructed_img.dtype}")
        return reconstructed_img

    
    def train_huffman_from_image(self, training_img, is_source_rgb=True):
        """
        Finds the symbols representing the image, extracts the 
        probability distribution of them and trains the huffman coder with it.

        training_img: np.array of shape [H, W, C]

        returns:
            Nothing
        """
        # Convert RGB to YCbCr if needed
        img_symbols = self.image2symbols(training_img, is_source_rgb)
        img_symbols = np.array(img_symbols, dtype=np.int32)  # Ensure consistent type

        safety_margin = 20
        self.bounds = (int(img_symbols.min()) - safety_margin, 
                    int(img_symbols.max()) + safety_margin + 1)        
        
        pmf = stats_marg(img_symbols, pixel_range=np.arange(self.bounds[0], self.bounds[1]))
        pmf = smooth_pmf(pmf)
        self.huffman = HuffmanCoder(lower_bound=self.bounds[0]) # still innit
        self.huffman.train(pmf)
        return None

    def intra_encode(self, img: np.array, return_bpp = False, is_source_rgb=True):
        """
        Encodes an image to a bitstream and return it by converting it to
        symbols and compressing them with the Huffman coder.

        img: np.array of shape [H, W, C]

        returns:
            bitstream: List of integers produced by the Huffman coder
        """
        symbols = self.image2symbols(img, is_source_rgb)
        bitstream, bitsize = self.huffman.encode(symbols)
        self.num_symbols = len(symbols)

        if return_bpp:
            total_pixels = img.shape[0] * img.shape[1]
            bpp = bitsize / total_pixels
            return bitstream, bpp
        else:
            return bitstream, None
    
    def intra_decode(self, bitstream, original_shape):
        """
        Decodes an image from a bitstream by decoding it with the Huffman
        coder and reconstructing it from the symbols.

        bitstream: List of integers produced by the Huffman coder
        original_shape: List of 3 values that contain H, W, and C

        returns:
            reconstructed_img: np.array of shape [H, W, C]

        """
        if not hasattr(self, 'num_symbols'):
            raise RuntimeError("No symbol count found. Make sure to encode first or store symbol count.")
        
        # Use the stored number of symbols instead of bitstream length
        decoded = self.huffman.decode(bitstream, self.num_symbols)
        reconstructed_img = self.symbols2image(decoded, original_shape)
        return reconstructed_img
    
    def encode_decode(self, img: np.array, return_bpp=False, is_source_rgb=True):
        """
        Encodes and then decodes an image using Huffman coding and quantization.

        img: np.array of shape [H, W, C]
        return_bpp: whether to return bits-per-pixel
        is_source_rgb: whether the input image is in RGB (converted to YCbCr)

        returns:
            reconstructed_img: np.array of shape [H, W, C]
            bitstream: List of integers produced by the Huffman coder
            bitsize: total number of bits used
        """
        # Encode image to bitstream
        symbols = self.image2symbols(img, is_source_rgb)
        bitstream, bitsize = self.huffman.encode(symbols)
        self.num_symbols = len(symbols)

        # Decode bitstream to image
        decoded = self.huffman.decode(bitstream, self.num_symbols)
        reconstructed_img = self.symbols2image(decoded, img.shape)

        if return_bpp:
            total_pixels = img.shape[0] * img.shape[1]
            bpp = bitsize / total_pixels
            return reconstructed_img, bitstream, bitsize, bpp
        else:
            return reconstructed_img, bitstream, bitsize
        

class IntraCodecAdaptive(IntraCodec):

    def _serialize_codebook(self):
        """
        Serialize the current Huffman codebook to bytes.
        You can store pmf or directly the Huffman table.
        Here we serialize pmf for simplicity.
        """
        # Get PMF from trained Huffman
        pmf = self.huffman.pmf
        # Serialize with pickle
        return pickle.dumps(pmf)

    def _deserialize_codebook(self, serialized_codebook):
        """
        Deserialize codebook bytes back to pmf and retrain HuffmanCoder
        """
        pmf = pickle.loads(serialized_codebook)
        self.huffman = HuffmanCoder(lower_bound=self.bounds[0])
        self.huffman.train(pmf)

    def intra_encode(self, img: np.array, return_bpp=False, is_source_rgb=True):
        """
        Overriding to retrain huffman per frame and output codebook + encoded symbols.
        """
        symbols = self.image2symbols(img, is_source_rgb)

        # Update bounds and train Huffman per frame
        symbols_arr = np.array(symbols, dtype=np.int32)
        safety_margin = 20
        self.bounds = (int(symbols_arr.min()) - safety_margin,
                       int(symbols_arr.max()) + safety_margin + 1)
        pmf = stats_marg(symbols_arr, pixel_range=np.arange(self.bounds[0], self.bounds[1]))
        pmf = smooth_pmf(pmf)
        self.huffman = HuffmanCoder(lower_bound=self.bounds[0])
        self.huffman.train(pmf)

        # Serialize codebook
        serialized_codebook = self._serialize_codebook()

        # Encode symbols
        bitstream, bitsize = self.huffman.encode(symbols)
        self.num_symbols = len(symbols)

        # Pack output: store codebook size + codebook bytes + bitstream
        codebook_len = len(serialized_codebook)
        # Store as tuple for clarity (you can serialize differently as needed)
        return (codebook_len, serialized_codebook, bitstream, self.num_symbols), bitsize


    def intra_decode(self, packed_bitstream, original_shape):
        """
        Decode from packed bitstream which includes the serialized codebook + encoded symbols.
        """
        codebook_len, serialized_codebook, bitstream, num_symbols = packed_bitstream
        self._deserialize_codebook(serialized_codebook)

        if not hasattr(self, 'num_symbols'):
            raise RuntimeError("No symbol count found. Make sure to encode first or store symbol count.")

        decoded = self.huffman.decode(bitstream, num_symbols)
        reconstructed_img = self.symbols2image(decoded, original_shape)
        return reconstructed_img
            
    
if __name__ == "__main__":
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt
    import numpy as np

    lena = imread(f'data/lena.tif')
    lena_small = imread(f'data/lena_small.tif')
    
    intracodec = IntraCodec(quantization_scale=0.15)
    intracodec.train_huffman_from_image(lena)
    
    bitstream, bpp = intracodec.intra_encode(lena, return_bpp=True)
    print(f"Bitstream length: {len(bitstream)}")
    print(f"Original symbol count: {intracodec.num_symbols}")
    
    reconstructed_img = intracodec.intra_decode(bitstream, lena.shape)
    psnr = calc_psnr(lena, reconstructed_img)
    print(f"PSNR: {psnr:.4f} dB, bpp: {bpp / (lena.size / 3)}")

    # ---------------------
    # Plot original vs reconstructed
    # ---------------------
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(np.clip(lena.astype(np.uint8), 0, 255))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB")
    plt.imshow(np.clip(reconstructed_img.astype(np.uint8), 0, 255))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

