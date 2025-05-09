import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.signal import rgb2gray
from scipy.stats import entropy
import numpy as np

def view_as_blocks(arr, block_shape):
    """
    Break a 2D array into non-overlapping blocks.

    Parameters:
        arr: np.ndarray
            Input 2D image array.
        block_shape: tuple
            Shape of each block (rows, cols).

    Returns:
        blocks: np.ndarray
            4D array of shape (num_blocks_y, num_blocks_x, block_rows, block_cols)
    """
    (h, w, _) = arr.shape
    bh, bw = block_shape
    assert h % bh == 0 and w % bw == 0, "Image dimensions must be divisible by block size"
    
    # Reshape and swap axes to get block view
    blocks = arr.reshape(h // bh, bh, w // bw, bw).swapaxes(1, 2)
    return blocks

# Load grayscale image
image = imread('data/lena.tif')
image = rgb2gray(image)
# image = (image * 255).astype(np.uint8)  # Ensure 8-bit range

# Divide into 16x16 blocks
block_size = 16
blocks = view_as_blocks(image, block_shape=(block_size, block_size))

# Initialize entropy map
entropy_map = np.zeros((blocks.shape[0], blocks.shape[1]))

# Compute entropy per block
for i in range(blocks.shape[0]):
    for j in range(blocks.shape[1]):
        block = blocks[i, j]
        hist, _ = np.histogram(block, bins=256, range=(0, 256), density=True)
        entropy_map[i, j] = entropy(hist, base=2)

# Visualize heatmap
plt.figure(figsize=(6, 6))
plt.imshow(entropy_map, cmap='hot', interpolation='nearest')
plt.title('Local Entropy Map (16x16 blocks)')
plt.colorbar(label='Entropy (bits)')
plt.axis('off')
plt.show()
