import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.entropy.zerorun import ZeroRunCoder
from ivclab.signal.dct import DiscreteCosineTransform
from ivclab.quantization.patchquant import PatchQuant
from ivclab.utils.shape import ZigZag
from ivclab.entropy.huffman import HuffmanCoder
from einops import rearrange

if __name__ == "__main__":
    # Load RGB image
    img_rgb = imread("data/lena_small.tif")

    # Convert RGB to YCbCr
    img_ycbcr = rgb2ycbcr(img_rgb)
    
    # Convert to float and center around 0 for DCT (subtract 128)
    img_ycbcr = img_ycbcr.astype(np.float32) - 128.0

    # Constants
    block_size = 8
    H, W, C = img_ycbcr.shape
    assert H % block_size == 0 and W % block_size == 0
    H_patch, W_patch = H // block_size, W // block_size

    # Patch image into 8x8 blocks
    patches = rearrange(img_ycbcr, '(h ph) (w pw) c -> h w c ph pw', ph=block_size, pw=block_size)
    print(f"Patches shape: {patches.shape}")

    # Initialize processors
    dct = DiscreteCosineTransform()
    quantizer = PatchQuant()
    zz = ZigZag()
    zr = ZeroRunCoder()

    # Process each 8x8 block individually
    all_encoded_symbols = []
    
    # Storage for processed patches (for visualization)
    dct_patches = np.zeros_like(patches, dtype=np.float32)
    quantized = np.zeros_like(patches, dtype=np.int32)
    
    print("Processing blocks...")
    
    for h in range(H_patch):
        for w in range(W_patch):
            # Extract all 3 channels for this spatial block position
            block_3ch = patches[h, w, :, :, :]  # Shape: (3, 8, 8)
            
            # 1. DCT - Apply to each channel separately
            dct_block_3ch = np.zeros_like(block_3ch, dtype=np.float32)
            for c in range(C):
                dct_block_3ch[c] = dct.transform(block_3ch[c])
            dct_patches[h, w, :, :, :] = dct_block_3ch
            
            # 2. Quantization - Your quantizer expects and returns 3-channel blocks
            quant_block_3ch = quantizer.quantize(dct_block_3ch)
            quantized[h, w, :, :, :] = quant_block_3ch
    
    # 3. Zig-Zag scan - Apply to all quantized patches at once
    # Your ZigZag expects shape (h, w, c, p0, p1)
    zz_scanned = zz.flatten(quantized)
    
    # 4. Zero-run encoding - Apply to all zig-zag scanned data at once
    # Your ZeroRunCoder expects shape (h, w, c, p) where p=64
    print("Applying zero-run encoding...")
    encoded_zr = zr.encode(zz_scanned)
    
    # Collect all encoded symbols
    all_encoded_symbols = list(encoded_zr.flatten())
    
    print(f"Total blocks processed: {H_patch * W_patch * C}")
    print(f"Total encoded symbols: {len(all_encoded_symbols)}")

    # 5. Huffman coding on all symbols
    if len(all_encoded_symbols) > 0:
        min_val = np.min(all_encoded_symbols)
        max_val = np.max(all_encoded_symbols)
        hist_range = max_val - min_val + 1
        histogram = np.zeros(hist_range, dtype=np.int32)
        
        for symbol in all_encoded_symbols:
            histogram[symbol - min_val] += 1
            
        pmf = histogram / np.sum(histogram)
        huffman = HuffmanCoder(lower_bound=min_val)
        huffman.train(pmf.astype(np.float32))
        huff_encoded, _ = huffman.encode(all_encoded_symbols)

        print(f"Image processed and Huffman table built.")
        print(f"Number of symbols encoded: {len(all_encoded_symbols)}")
        print(f"Unique symbols: {np.unique(all_encoded_symbols).shape[0]}")

    # ====== Visualization ======
    plt.figure(figsize=(15, 5))

    # Show Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Show quantized Y channel
    plt.subplot(1, 3, 2)
    quant_Y = quantized[:, :, 0, :, :]  # Only Y channel
    quant_Y_full = rearrange(quant_Y, 'h w ph pw -> (h ph) (w pw)')
    plt.imshow(quant_Y_full, cmap='gray', vmin=-50, vmax=50)
    plt.title("Quantized Y Channel")
    plt.axis("off")

    # Show DCT coefficients of Y channel (for debugging)
    plt.subplot(1, 3, 3)
    dct_Y = dct_patches[:, :, 0, :, :]  # Only Y channel DCT
    dct_Y_full = rearrange(dct_Y, 'h w ph pw -> (h ph) (w pw)')
    plt.imshow(dct_Y_full, cmap='gray', vmin=-200, vmax=200)
    plt.title("DCT Coefficients Y Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Additional statistics
    print(f"\nStatistics:")
    print(f"Original image shape: {img_rgb.shape}")
    print(f"YCbCr image shape: {img_ycbcr.shape}")
    print(f"Number of 8x8 blocks per channel: {H_patch} x {W_patch} = {H_patch * W_patch}")
    print(f"Total number of blocks (all channels): {H_patch * W_patch * C}")
    
    # Check if any quantized coefficients are non-zero (good sign)
    non_zero_coeffs = np.count_nonzero(quantized)
    total_coeffs = quantized.size
    print(f"Non-zero quantized coefficients: {non_zero_coeffs}/{total_coeffs} ({100*non_zero_coeffs/total_coeffs:.1f}%)")