import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt


class VideoCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8),
                 search_range = 4
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound= -((2*search_range + 1)**2 - 1)//2)

        self.decoder_recon = None
    
    def encode_decode(self, frame, frame_num=0, is_source_rgb=False):
        # Explanation:
        # For frame 0: pure intra coding using your IntraCodec.

        # For other frames:
        # - Motion vector is computed block-wise using SSD.
        # - Motion vector is Huffman encoded after computing its histogram.
        # - Prediction is reconstructed using motion compensation.
        # - Residual is coded via the same intra pipeline.
        # - Final reconstruction is prediction + decoded residual.

        # Assumptions:
        # - You already implemented and validated the IntraCodec, MotionCompensator, HuffmanCoder, rgb2ycbcr, ycbcr2rgb.
        # - Frame input is in standard uint8 RGB format.
        # - Decoder stores self.decoder_recon to use for the next frame's motion estimation.
        # Convert RGB to YCbCr
        frame_ycbcr = rgb2ycbcr(frame.astype(np.float32))
        y_channel = frame_ycbcr[..., 0]

        if frame_num == 0:
            # Intra-frame coding (I-frame)
            recon_y, bitstream, residual_bitsize = self.intra_codec.encode_decode(y_channel, is_source_rgb=is_source_rgb)
            motion_bitsize = 0
            self.decoder_recon = recon_y
        else:
            # Motion estimation
            motion_vector = self.motion_comp.compute_motion_vector(self.decoder_recon, y_channel)

            # Encode motion vector using Huffman
            flat_mv = motion_vector.flatten()
            hist_mv = stats_marg(flat_mv)
            self.motion_huffman.train(hist_mv)
            motion_encoded, motion_bitsize = self.motion_huffman.encode(flat_mv)

            # Motion compensated prediction
            prediction = self.motion_comp.reconstruct_with_motion_vector(self.decoder_recon[..., np.newaxis], motion_vector)[..., 0]

            # Compute residual
            residual = y_channel - prediction

            # Encode residual
            recon_residual, bitstream, residual_bitsize = self.residual_codec.encode_decode(residual, is_source_rgb=is_source_rgb)

            # Reconstruct current frame from prediction + residual
            recon_y = prediction + recon_residual
            self.decoder_recon = recon_y

        # Merge reconstructed Y channel with original CbCr
        recon_frame_ycbcr = frame_ycbcr.copy()
        recon_frame_ycbcr[..., 0] = np.clip(recon_y, 0, 255)

        # Convert YCbCr back to RGB
        recon_rgb = ycbcr2rgb(recon_frame_ycbcr).astype(np.uint8)

        bitsize = residual_bitsize + motion_bitsize
        return recon_rgb, bitstream, bitsize

