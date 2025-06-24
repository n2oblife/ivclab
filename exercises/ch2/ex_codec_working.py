import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.signal import rgb2ycbcr, ycbcr2rgb, downsample, interpolation_upsample
from ivclab.entropy.huffman import HuffmanCoder
from ivclab.utils.metrics import calc_psnr
from ivclab.entropy import calc_entropy

# Fixed 3-pixel predictor coefficients
coefficients = [7/8, -4/8, 5/8]

def three_pixel_predictor_residual(img_channel):
    """
    Compute prediction residuals using 3-pixel predictor during encoding.
    Uses original image pixels for prediction.
    """
    H, W = img_channel.shape
    img_channel = img_channel.astype(np.float32)
    residual = np.zeros_like(img_channel, dtype=np.float32)
    
    # First row and column: store original values as residuals
    residual[0, :] = img_channel[0, :]
    residual[:, 0] = img_channel[:, 0]

    for i in range(1, H):
        for j in range(1, W):
            # Use ORIGINAL pixels for prediction during encoding
            left = img_channel[i, j-1]
            top = img_channel[i-1, j]
            top_left = img_channel[i-1, j-1]
            prediction = coefficients[0]*left + coefficients[1]*top + coefficients[2]*top_left
            residual[i, j] = img_channel[i, j] - prediction

    return residual.astype(np.int16)

def reconstruct_from_residual(residual):
    """
    Reconstruct image from residuals using sequential pixel-by-pixel reconstruction.
    Uses previously reconstructed pixels for prediction (causal reconstruction).
    """
    H, W = residual.shape
    reconstruction = np.zeros_like(residual, dtype=np.float32)
    
    # First row and column: copy residuals directly (they contain original values)
    reconstruction[0, :] = residual[0, :].astype(np.float32)
    reconstruction[:, 0] = residual[:, 0].astype(np.float32)

    for i in range(1, H):
        for j in range(1, W):
            # Use RECONSTRUCTED pixels for prediction during decoding
            left = reconstruction[i, j-1]
            top = reconstruction[i-1, j]
            top_left = reconstruction[i-1, j-1]
            prediction = coefficients[0]*left + coefficients[1]*top + coefficients[2]*top_left
            reconstruction[i, j] = prediction + residual[i, j]

    return np.clip(reconstruction, 0, 255).astype(np.uint8)

def build_huffman_codebook_from_reference():
    """
    Build Huffman codebook from lena_small.tif reference image.
    """
    try:
        # Try to load actual lena_small.tif
        ref_img = imread("data/lena_small.tif")
        ref_ycbcr = rgb2ycbcr(ref_img)
        ref_Y = ref_ycbcr[:, :, 0]
    except:
        # Fallback: simulate lena_small.tif with more realistic statistics
        print("Warning: Using simulated reference image for Huffman training")
        np.random.seed(42)  # Fixed seed for reproducibility
        # Create more realistic image with spatial correlation
        ref_Y = np.random.normal(128, 40, (64, 64))  # More realistic intensity distribution
        ref_Y = np.clip(ref_Y, 0, 255).astype(np.uint8)
        
        # Add some spatial structure
        from scipy.ndimage import gaussian_filter
        ref_Y = gaussian_filter(ref_Y.astype(np.float32), sigma=1.0)
        ref_Y = np.clip(ref_Y, 0, 255).astype(np.uint8)
    
    # Compute residuals from reference image
    ref_residual = three_pixel_predictor_residual(ref_Y)
    ref_flat = ref_residual.flatten()
    
    # Build histogram with extended range to handle all possible residuals
    min_val, max_val = -512, 512  # Extended range for safety
    histogram = np.zeros(max_val - min_val + 1, dtype=np.int64)
    
    for val in ref_flat:
        idx = int(val - min_val)
        if 0 <= idx < len(histogram):
            histogram[idx] += 1
    
    # Add smoothing to ensure no zero probabilities
    histogram += 1
    pmf = histogram / np.sum(histogram)
    
    # Train Huffman coder
    huffman = HuffmanCoder(lower_bound=min_val)
    huffman.train(pmf.astype(np.float32))
    
    return huffman, ref_residual

def encode_image(img_rgb, huffman_coder):
    """
    Complete image encoding pipeline.
    """
    # Step 1: RGB to YCbCr conversion
    ycbcr = rgb2ycbcr(img_rgb)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    
    # Step 2: Chroma subsampling (4:2:0)
    Cb_ds = downsample(Cb, 2)
    Cr_ds = downsample(Cr, 2)
    
    # Step 3: Compute prediction residuals
    residual_Y = three_pixel_predictor_residual(Y)
    
    # For chroma: simple DC removal (subtract 128)
    residual_Cb = Cb_ds.astype(np.int16) - 128
    residual_Cr = Cr_ds.astype(np.int16) - 128
    
    # Step 4: Huffman encode all channels
    channels = [residual_Y, residual_Cb, residual_Cr]
    compressed_streams = []
    shapes = []
    total_bits = 0
    
    for i, channel in enumerate(channels):
        compressed, _ = huffman_coder.encode(channel.flatten())
        compressed_streams.append(compressed)
        shapes.append(channel.shape)
        # Estimate bits (this is approximate - actual implementation would track exact bits)
        total_bits += len(compressed) * 32  # 32 bits per int approximation
    
    return compressed_streams, shapes, total_bits

def decode_image(compressed_streams, shapes, huffman_coder, original_shape):
    """
    Complete image decoding pipeline.
    """
    # Step 1: Huffman decode all channels
    decoded_Y = huffman_coder.decode(compressed_streams[0], np.prod(shapes[0])).reshape(shapes[0])
    decoded_Cb = huffman_coder.decode(compressed_streams[1], np.prod(shapes[1])).reshape(shapes[1])
    decoded_Cr = huffman_coder.decode(compressed_streams[2], np.prod(shapes[2])).reshape(shapes[2])
    
    # Step 2: Reconstruct Y channel using sequential prediction
    reconstructed_Y = reconstruct_from_residual(decoded_Y)
    
    # Step 3: Reconstruct chroma channels
    reconstructed_Cb = np.clip(decoded_Cb + 128, 0, 255).astype(np.uint8)
    reconstructed_Cr = np.clip(decoded_Cr + 128, 0, 255).astype(np.uint8)
    
    # Step 4: Upsample chroma channels
    Cb_us = interpolation_upsample(reconstructed_Cb, 2)
    Cr_us = interpolation_upsample(reconstructed_Cr, 2)
    
    # Ensure upsampled chroma matches Y dimensions
    H, W = reconstructed_Y.shape
    Cb_us = Cb_us[:H, :W]  # Crop if necessary
    Cr_us = Cr_us[:H, :W]
    
    # Step 5: Combine YCbCr and convert back to RGB
    rec_ycbcr = np.stack([reconstructed_Y, Cb_us, Cr_us], axis=2)
    img_rgb_rec = np.clip(ycbcr2rgb(rec_ycbcr), 0, 255).astype(np.uint8)
    
    return img_rgb_rec

if __name__ == "__main__":
    # Load test image
    img_rgb = imread("data/lena.tif")
    print(f"Original image shape: {img_rgb.shape}")
    
    # Build Huffman codebook from reference image
    print("Building Huffman codebook from reference image...")
    huffman_coder, ref_residual = build_huffman_codebook_from_reference()
    print(f"Reference residual range: [{ref_residual.min()}, {ref_residual.max()}]")
    
    # Encode image
    print("Encoding image...")
    compressed_streams, shapes, total_bits = encode_image(img_rgb, huffman_coder)
    
    # Decode image
    print("Decoding image...")
    img_rgb_rec = decode_image(compressed_streams, shapes, huffman_coder, img_rgb.shape)
    
    # Calculate metrics
    psnr = calc_psnr(img_rgb, img_rgb_rec)
    bpp = total_bits / (img_rgb.shape[0] * img_rgb.shape[1])
    
    # Calculate entropy of reconstructed image
    hist, _ = np.histogram(img_rgb_rec.flatten(), bins=256, range=(0, 256))
    pmf_rec = hist / np.sum(hist)
    pmf_rec = pmf_rec[pmf_rec > 0]  # Remove zeros for entropy calculation
    entropy_rec = calc_entropy(pmf_rec)
    
    original_bits = img_rgb.size * 8
    compression_ratio = original_bits / total_bits
    
    # Print results
    print(f"\n=== COMPRESSION RESULTS ===")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bits per pixel: {bpp:.3f}")
    print(f"Entropy of reconstructed image: {entropy_rec:.3f} bits/pixel")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")
    print(f"Original size: {original_bits} bits")
    print(f"Compressed size: {total_bits} bits")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_rgb_rec)
    plt.title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    error_img = np.abs(img_rgb.astype(np.float32) - img_rgb_rec.astype(np.float32))
    plt.imshow(error_img.astype(np.uint8))
    plt.title("Absolute Error")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Show residual statistics
    ycbcr = rgb2ycbcr(img_rgb)
    Y = ycbcr[:, :, 0]
    residual_Y = three_pixel_predictor_residual(Y)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residual_Y.flatten(), bins=50, alpha=0.7, density=True)
    plt.title("Residual Distribution (Y channel)")
    plt.xlabel("Residual Value")
    plt.ylabel("Density")
    
    plt.subplot(1, 2, 2)
    plt.hist(ref_residual.flatten(), bins=50, alpha=0.7, density=True, color='orange')
    plt.title("Reference Residual Distribution")
    plt.xlabel("Residual Value")
    plt.ylabel("Density")
    
    plt.tight_layout()
    plt.show()