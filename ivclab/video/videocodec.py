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
    
    # def encode_decode(self, frame, frame_num=0):

    #     if frame_num == 0:
    #     # YOUR CODE STARTS HERE
    #     # raise NotImplementedError()
    #     # YOUR CODE ENDS HERE
    #     bitsize = residual_bitsize + motion_bitsize
    #     return self.decoder_recon, bitstream, bitsize
