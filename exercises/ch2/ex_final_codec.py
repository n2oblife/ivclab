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
    H, W = img_channel.shape
    img_channel = img_channel.astype(np.float32)
    reconstruction = np.zeros_like(img_channel)
    reconstruction[0, :] = img_channel[0, :]
    reconstruction[:, 0] = img_channel[:, 0]
    residual = np.zeros_like(img_channel)

    for i in range(1, H):
        for j in range(1, W):
            left = reconstruction[i, j-1]
            top = reconstruction[i-1, j]
            top_left = reconstruction[i-1, j-1]
            prediction = coefficients[0]*left + coefficients[1]*top + coefficients[2]*top_left
            residual[i, j] = img_channel[i, j] - prediction
            reconstruction[i, j] = prediction + residual[i, j]

    return residual.astype(np.int16)

def reconstruct_from_residual(residual):
    H, W = residual.shape
    reconstruction = np.zeros_like(residual, dtype=np.float32)
    reconstruction[0, :] = residual[0, :]
    reconstruction[:, 0] = residual[:, 0]

    for i in range(1, H):
        for j in range(1, W):
            left = reconstruction[i, j-1]
            top = reconstruction[i-1, j]
            top_left = reconstruction[i-1, j-1]
            prediction = coefficients[0]*left + coefficients[1]*top + coefficients[2]*top_left
            reconstruction[i, j] = prediction + residual[i, j]

    return np.clip(reconstruction, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    # Step 1: Load test image
    img_rgb = imread("data/lena.tif")  # Your actual image to encode
    ycbcr = rgb2ycbcr(img_rgb)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    # Step 2: Chroma subsampling
    Cb_ds = downsample(Cb, 2)
    Cr_ds = downsample(Cr, 2)

    # Step 3: Predict and get residuals
    # Quantization factors to simulate bitrate control (more = stronger compression)
    quant_levels = [1, 2, 4, 8, 16, 32, 64]
    psnr_list = []
    bpp_list = []

    for q in quant_levels:
        print(f"Processing quantization factor: {q}")

        # Quantize residuals
        q_residual_Y = (residual_Y // q).astype(np.int16)
        q_residual_Cb = (residual_Cb // q).astype(np.int16)
        q_residual_Cr = (residual_Cr // q).astype(np.int16)

        # Encode
        total_bits = 0
        compressed_streams = []
        shapes = []

        for channel in [q_residual_Y, q_residual_Cb, q_residual_Cr]:
            compressed, _ = huffman.encode(channel.flatten())
            compressed_streams.append(compressed)
            shapes.append(channel.shape)
            total_bits += len(compressed) * 32  # 32 bits per encoded int

        # Decode
        dq_residual_Y = huffman.decode(compressed_streams[0], np.prod(shapes[0])).reshape(shapes[0]) * q
        dq_residual_Cb = huffman.decode(compressed_streams[1], np.prod(shapes[1])).reshape(shapes[1]) * q + 128
        dq_residual_Cr = huffman.decode(compressed_streams[2], np.prod(shapes[2])).reshape(shapes[2]) * q + 128

        # Reconstruct
        rec_Y = reconstruct_from_residual(dq_residual_Y)
        rec_Cb_us = interpolation_upsample(np.clip(dq_residual_Cb, 0, 255).astype(np.uint8), 2)
        rec_Cr_us = interpolation_upsample(np.clip(dq_residual_Cr, 0, 255).astype(np.uint8), 2)

        rec_ycbcr = np.stack([rec_Y, rec_Cb_us, rec_Cr_us], axis=2)
        rec_rgb = np.clip(ycbcr2rgb(rec_ycbcr), 0, 255).astype(np.uint8)

        # Metrics
        psnr = calc_psnr(img_rgb, rec_rgb)
        bpp = total_bits / (img_rgb.shape[0] * img_rgb.shape[1])

        psnr_list.append(psnr)
        bpp_list.append(bpp)

        print(f"  -> PSNR: {psnr:.2f} dB, BPP: {bpp:.3f}")


    # Step 4: Simulate reference image (lena_small.tif) and build Huffman model
    np.random.seed(0)
    ref_Y = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    ref_residual = three_pixel_predictor_residual(ref_Y)
    ref_flat = ref_residual.flatten()

    # Pad histogram to cover full range [-255, 255]
    min_val, max_val = -255, 255
    histogram = np.zeros(max_val - min_val + 1, dtype=np.int64)
    for val in ref_flat:
        idx = int(val - min_val)
        if 0 <= idx < len(histogram):
            histogram[idx] += 1
    histogram += 1  # smoothing to ensure no zero probabilities
    pmf = histogram / np.sum(histogram)

    huffman = HuffmanCoder(lower_bound=min_val)
    huffman.train(pmf.astype(np.float32))

    # Step 5: Encode residuals
    channels = [residual_Y, residual_Cb, residual_Cr]
    compressed_streams, shapes, total_bits = [], [], 0

    for channel in channels:
        compressed, _ = huffman.encode(channel.flatten())
        compressed_streams.append(compressed)
        shapes.append(channel.shape)
        total_bits += len(compressed) * 32  # 32 bits per compressed int

    # Step 6: Decode and reconstruct Y
    decoded_Y = huffman.decode(compressed_streams[0], np.prod(shapes[0])).reshape(shapes[0])
    reconstructed_Y = reconstruct_from_residual(decoded_Y)

    # Step 7: Decode and reconstruct chroma
    decoded_Cb = huffman.decode(compressed_streams[1], np.prod(shapes[1])).reshape(shapes[1]) + 128
    decoded_Cr = huffman.decode(compressed_streams[2], np.prod(shapes[2])).reshape(shapes[2]) + 128

    Cb_us = interpolation_upsample(np.clip(decoded_Cb, 0, 255).astype(np.uint8), 2)
    Cr_us = interpolation_upsample(np.clip(decoded_Cr, 0, 255).astype(np.uint8), 2)

    # Step 8: Recombine and convert back to RGB
    rec_ycbcr = np.stack([reconstructed_Y, Cb_us, Cr_us], axis=2)
    img_rgb_rec = np.clip(ycbcr2rgb(rec_ycbcr), 0, 255).astype(np.uint8)

    # Metrics
    psnr = calc_psnr(img_rgb, img_rgb_rec)
    bpp = total_bits / (img_rgb.shape[0] * img_rgb.shape[1])
    hist, _ = np.histogram(img_rgb_rec.flatten(), bins=256, range=(0, 256))
    pmf = hist / np.sum(hist)
    entropy = calc_entropy(pmf)
    original_bits = img_rgb.size * 8
    compression_ratio = original_bits / total_bits

    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bits per pixel: {bpp:.3f}")
    print(f"Entropy after reconstruction: {entropy:.3f}bits/pixel")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")

    # Plot Rate-Distortion Curve
    plt.figure()
    plt.plot(bpp_list, psnr_list, marker='o')
    plt.title("Rate-Distortion Curve")
    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_rec)
    plt.title("Reconstructed Image")
    plt.show()
