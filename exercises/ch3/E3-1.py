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

    # Constants
    block_size = 8
    H, W, C = img_ycbcr.shape
    assert H % block_size == 0 and W % block_size == 0
    H_patch, W_patch = H // block_size, W // block_size

    # Patch image into 8x8 blocks
    patches = rearrange(img_ycbcr, '(h ph) (w pw) c -> h w c ph pw', ph=block_size, pw=block_size)

    # 1. DCT
    dct = DiscreteCosineTransform()
    dct_patches = dct.transform(patches)

    # 2. Quantization
    quantizer = PatchQuant()
    quantized = quantizer.quantize(dct_patches)

    # 3. Zig-Zag scan
    zz = ZigZag()
    zz_scanned = zz.flatten(quantized)

    # 4. Zero-run encoding
    zr = ZeroRunCoder()
    encoded_zr = zr.encode(zz_scanned)

    # 5. Huffman coding (fully robust)
    symbols, counts = np.unique(encoded_zr, return_counts=True)
    pmf = counts / np.sum(counts)

    # Map original symbols to indices (0-based)
    symbol_to_index = {s: i for i, s in enumerate(symbols)}
    indexed_encoded_zr = np.array([symbol_to_index[s] for s in encoded_zr], dtype=np.int32)

    # Train Huffman
    huffman = HuffmanCoder(lower_bound=0)  # Since we mapped to 0-based indices
    huffman.train(pmf.astype(np.float32))

    # Encode
    huff_encoded, bitstring = huffman.encode(indexed_encoded_zr)


    print(f"Image processed and Huffman table built.")
    print(f"Number of symbols encoded: {len(encoded_zr)}")
    print(f"Unique symbols: {np.unique(encoded_zr).shape[0]}")

    # ====== Visualization ======
    # Show Original Image
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("lena Image")
    plt.axis("off")

    # Show quantized Y channel as encoded "visual"
    quant_Y = quantized[:, :, 0, :, :]  # Only Y channel
    quant_Y_full = rearrange(quant_Y, 'h w ph pw -> (h ph) (w pw)')
    plt.subplot(1, 2, 2)
    plt.imshow(quant_Y_full, cmap='gray', vmin=-50, vmax=50)
    plt.title("Quantized Y Channel (as Encoded)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
