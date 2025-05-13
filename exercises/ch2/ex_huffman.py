import numpy as np
from PIL import Image
from collections import Counter
from heapq import heappush, heappop, heapify
import matplotlib.pyplot as plt

# Load 'lena_small.tif' image (simulate since we can't read files)
# We'll create a mock grayscale image as a substitute
np.random.seed(0)
lena_small = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)  # 64x64 grayscale mock

# Minimum-entropy predictor coefficients (3-pixel predictor)
coefficients = [7 / 8, -4 / 8, 5 / 8]

def min_entropy_predictor(image, coefficients):
    H, W = image.shape
    image = image.astype(np.float32)
    
    reconstruction = np.zeros_like(image)
    reconstruction[0, :] = image[0, :]
    reconstruction[:, 0] = image[:, 0]
    
    residual_error = np.copy(reconstruction)

    for i in range(1, H):
        for j in range(1, W):
            left = reconstruction[i, j - 1]
            top = reconstruction[i - 1, j]
            top_left = reconstruction[i - 1, j - 1]
            prediction = (
                coefficients[0] * left +
                coefficients[1] * top +
                coefficients[2] * top_left
            )
            actual = image[i, j]
            error = np.round(actual - prediction)
            residual_error[i, j] = error
            reconstruction[i, j] = prediction + error

    return residual_error.astype(np.int16)

# Generate residuals
residuals = min_entropy_predictor(lena_small, coefficients)

# Flatten and count occurrences of all possible values in range [-255, 255]
residuals_flat = residuals.flatten()
residual_range = np.arange(-255, 256)
counts = Counter(residuals_flat)
probabilities = {val: counts.get(val, 0) / len(residuals_flat) for val in residual_range}

# Huffman Tree Implementation
class HuffmanNode:
    def __init__(self, symbol=None, prob=0):
        self.symbol = symbol
        self.prob = prob
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.prob < other.prob

def build_huffman_tree(probabilities):
    heap = [HuffmanNode(symbol=s, prob=p) for s, p in probabilities.items()]
    heapify(heap)

    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = HuffmanNode(prob=node1.prob + node2.prob)
        merged.left = node1
        merged.right = node2
        heappush(heap, merged)

    return heap[0]

def generate_codes(node, prefix='', codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_codes(node.left, prefix + '0', codebook)
        generate_codes(node.right, prefix + '1', codebook)
    return codebook

# Build and generate Huffman code
huffman_tree = build_huffman_tree(probabilities)
codebook = generate_codes(huffman_tree)

# Compute codeword lengths
codeword_lengths = {symbol: len(code) for symbol, code in codebook.items()}

# Results
num_codewords = len(codebook)
max_codeword_length = max(codeword_lengths.values())
min_codeword_length = min(codeword_lengths.values())

# Plot codelengths
symbols = list(codeword_lengths.keys())
lengths = [codeword_lengths[s] for s in symbols]

plt.figure(figsize=(12, 6))
plt.bar(symbols, lengths)
plt.title("Huffman Codeword Lengths for Residuals")
plt.xlabel("Residual Error Value")
plt.ylabel("Codeword Length (bits)")
plt.grid(True)
plt.tight_layout()
plt.show()

(num_codewords, max_codeword_length, min_codeword_length)
print("Number of codewords:", num_codewords)
print("Max. codewordlength:", max_codeword_length)
print("Min. codewordlength:", min_codeword_length)