import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec, IntraCodecAdaptive
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt
import pickle


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

        # Intra codec for I-frame & residual
        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)

        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound= -((2*search_range + 1)**2 - 1)//2)

        self.decoder_recon = None
    
    def encode_decode(self, frame, frame_num=0, is_source_rgb=False):
        frame_ycbcr = rgb2ycbcr(frame.astype(np.float32))
        y_channel = frame_ycbcr[..., 0]

        if frame_num == 0:
            # I-frame: train Huffman on current Y channel symbols
            self.intra_codec.train_huffman_from_image(y_channel, is_source_rgb=False)

            # Encode/decode using the updated Huffman
            recon_y, bitstream, residual_bitsize = self.intra_codec.encode_decode(y_channel, is_source_rgb=False)
            motion_bitsize = 0
            self.decoder_recon = recon_y
        else:
            # Motion estimation
            decoder_y = self.decoder_recon[..., 0] if self.decoder_recon.ndim == 3 else self.decoder_recon
            motion_vector = self.motion_comp.compute_motion_vector(decoder_y, y_channel)

            flat_mv = motion_vector.flatten()

            # Train Huffman on full MV range (once)
            if frame_num == 1:  # or use frame_num == 0 but already used for intra
                num_symbols = (2 * self.motion_comp.search_range + 1) ** 2
                uniform_pmf = np.full(num_symbols, 1.0 / num_symbols)
                self.motion_huffman.train(uniform_pmf)

            motion_encoded, motion_bitsize = self.motion_huffman.encode(flat_mv)
            decoded_mv = self.motion_huffman.decode(motion_encoded, len(flat_mv))
            motion_vector_decoded = decoded_mv.reshape(motion_vector.shape)

            # Motion compensation prediction
            decoder_y_expanded = decoder_y[..., np.newaxis]
            prediction = self.motion_comp.reconstruct_with_motion_vector(decoder_y_expanded, motion_vector_decoded)[..., 0]

            # Residual + Huffman encoding per frame
            residual = y_channel - prediction
            self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False)
            recon_residual, bitstream, residual_bitsize = self.residual_codec.encode_decode(residual, is_source_rgb=False)
            recon_y = prediction + recon_residual
            self.decoder_recon = recon_y

        # Combine reconstructed Y with original CbCr channels
        recon_frame_ycbcr = frame_ycbcr.copy()
        recon_y_channel = recon_y[..., 0] if recon_y.ndim == 3 else recon_y
        recon_frame_ycbcr[..., 0] = np.clip(recon_y_channel, 0, 255)

        recon_rgb = ycbcr2rgb(recon_frame_ycbcr).astype(np.uint8)

        # Combine bitstream of motion and residual (bitstream is just residual here)
        bitsize = residual_bitsize + motion_bitsize
        return recon_rgb, bitstream, bitsize




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