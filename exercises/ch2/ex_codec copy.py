import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.signal import rgb2ycbcr, ycbcr2rgb, downsample, upsample
from ivclab.entropy.huffman import HuffmanCoder
from ivclab.utils.metrics import calc_psnr
from ivclab.entropy import calc_entropy
from ivclab.image.predictive import three_pixels_predictor, _predict_from_neighbors
from scipy.signal import decimate



if __name__ == "__main__":
    # Load image
    img_rgb = imread("data/lena_small.tif")

    # Encode with Huffman
    # Step 1: Get residuals from 3-pixel predictor with chroma subsampling
    # downsampled = downsample(img_rgb, 2)
    # residual_Y, residual_CbCr = three_pixels_predictor(downsampled)
    residual_Y, residual_CbCr = three_pixels_predictor(img_rgb, subsample_color_channels=True)

    # Step 2: Flatten all residuals for Huffman coding
    residuals = [residual_Y, residual_CbCr[:,:,0], residual_CbCr[:,:,1]]
    all_residuals_concat = np.concatenate([r.flatten() for r in residuals])

    # Step 3: Train HuffmanCoder
    min_val = np.min(all_residuals_concat)
    max_val = np.max(all_residuals_concat)
    hist_range = max_val - min_val + 1
    histogram = np.zeros(hist_range, dtype=np.int64)
    for val in all_residuals_concat:
        histogram[val - min_val] += 1
    pmf = histogram / np.sum(histogram)
    huffman = HuffmanCoder(lower_bound=min_val)
    huffman.train(pmf.astype(np.float32))

    # Step 4: Encode all residuals
    compressed_streams = []
    bitrates = []
    total_bits = 0
    shapes = []

    for residual in residuals:
        compressed, bitrate = huffman.encode(residual.flatten())
        compressed_streams.append(compressed)
        bitrates.append(bitrate)
        total_bits += len(compressed) * 32  # 32 bits per compressed word
        shapes.append(residual.shape)

    # Step 5: Decode residuals
    decoded_residuals = []
    for i, compressed in enumerate(compressed_streams):
        decoded = huffman.decode(compressed, np.prod(shapes[i])) # flat
        decoded_residuals.append(decoded.reshape(shapes[i])) # reshaped

    # Step 6: Reconstruct (Y and CbCr) from decoding and prediction
    reconstructed = []
    for i in range(len(residuals)):
        reconstructed.append( np.clip(residuals[i] + decoded_residuals[i], 0, 255).astype(np.uint8) )

    # Step 7: Upsample chroma and reconstruct final YCbCr
    for i in range(1, len(reconstructed)):
        reconstructed[i] = upsample(reconstructed[i], 2)
    reconstructed = np.stack(reconstructed, axis=2)

    # Step 8: Convert to RGB
    img_rgb_rec = np.clip(ycbcr2rgb(reconstructed), 0, 255).astype(np.uint8)

    # Compute metrics
    psnr = calc_psnr(img_rgb, img_rgb_rec)
    entropy = calc_entropy(np.array(img_rgb_rec, dtype=np.uint8))
    bpp = total_bits / (img_rgb.shape[0] * img_rgb.shape[1])
    original_bits = img_rgb.size * 8  # Total pixels * 8 bits per channel
    compression_ratio = original_bits / total_bits

    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bits per pixel (Huffman stream): {bpp:.3f}")
    print(f"Estimated entropy (after reconstruction): {entropy:.3f} bits/pixel")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")

    # Optional: Show original and reconstructed
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_rec)
    plt.title("Reconstructed")

    plt.show()
