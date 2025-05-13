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
# # raise NotImplementedError()
# YOUR CODE ENDS HERE

all_bpps = np.array(all_bpps)
all_PSNRs = np.array(all_PSNRs)

print(all_bpps, all_PSNRs)
plt.plot(all_bpps, all_PSNRs, marker='o')
plt.show()
