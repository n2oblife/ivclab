import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.signal import rgb2ycbcr, ycbcr2rgb, downsample, upsample, interpolation_upsample
from ivclab.entropy.huffman import HuffmanCoder
from ivclab.utils.metrics import calc_psnr
from ivclab.entropy import calc_entropy
from ivclab.image.predictive import three_pixels_predictor

def train_huffman(img_rgb:np.ndarray)->HuffmanCoder:
    "Train a Huffman encoder based on an rgb image"
    # Step 1: Get residuals from 3-pixel predictor with chroma subsampling
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
    return huffman, residual_Y, residual_CbCr

def huffman_encoding(message, encoder:HuffmanCoder):
    # Step 4: Encode all residuals

    if isinstance(message, list):
        compressed_streams = []
        bitrates = []
        total_bits = 0
        shapes = []
        for residual in message:
            compressed, bitrate = encoder.encode(residual.flatten())
            compressed_streams.append(compressed)
            bitrates.append(bitrate)
            total_bits += len(compressed) * 32  # 32 bits per compressed word
            shapes.append(residual.shape)
        return compressed_streams, bitrates, total_bits, shapes
    else:
        compressed, bitrate = encoder.encode(message.flatten())
        return compressed, bitrate, len(compressed) * 32, message.shape