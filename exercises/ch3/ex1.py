import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread,calc_psnr
import matplotlib.pyplot as plt

# Implement the IntraCodec and all the necessary modules
# For each given quantization scale in the handout:
# - Initialize a new IntraCodec
# - Use lena_small to train Huffman coder of IntraCodec.
# - Compress and decompress 'lena.tif'
# - Measure bitrate and PSNR on lena
# Plot all the measurements in a Rate Distortion plot

lena = imread(f'data/lena.tif')
lena_small = imread(f'data/lena_small.tif')
H, W, C = lena.shape
all_PSNRs = list()
all_bpps = list()

# YOUR CODE STARTS HERE
quantization_scales = [0.05, 0.1, 0.15, 0.2, 0.3]

for q in quantization_scales:
    intracodec = IntraCodec(quantization_scale=q)
    intracodec.train_huffman_from_image(lena_small, is_source_rgb=False)

    # Encode image
    bitstream = intracodec.intra_encode(lena, return_bpp=False)
    # Compute bitsize (length of bitstream in bits)
    bitsize = len(bitstream)

    # Decode image
    reconstructed_img = intracodec.intra_decode(bitstream, lena.shape)

    # Calculate PSNR
    psnr = calc_psnr(lena, reconstructed_img)
    all_PSNRs.append(psnr)

    # Calculate bitrate in bits per pixel
    bpp = bitsize / (H * W)
    all_bpps.append(bpp)

    print(f"Quant Scale: {q} | PSNR: {psnr:.2f} dB | bpp: {bpp:.4f}")
# YOUR CODE ENDS HERE

all_bpps = np.array(all_bpps)
all_PSNRs = np.array(all_PSNRs)

print(all_bpps, all_PSNRs)
plt.plot(all_bpps, all_PSNRs, marker='o')
plt.show()
