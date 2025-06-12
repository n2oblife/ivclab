import numpy as np
from collections import Counter
from heapq import heappush, heappop, heapify
import matplotlib.pyplot as plt

# Simulated grayscale image as placeholder for 'lena_small.tif'
np.random.seed(0)
lena_small = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)

# Minimum-entropy predictor
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

# Step 1: Get residuals
residuals = min_entropy_predictor(lena_small, coefficients)
residuals_flat = residuals.flatten()

# Step 2: Calculate probabilities
residual_range = np.arange(-255, 256)
counts = Counter(residuals_flat)
total = len(residuals_flat)
probabilities = {val: counts.get(val, 0) / total for val in residual_range}

# Step 3: Remove zero-probability symbols for tree building
nonzero_probs = {s: p for s, p in probabilities.items() if p > 0}

# Step 4: Huffman Tree
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

# Step 5: Generate codes
huffman_tree = build_huffman_tree(nonzero_probs)
codebook = generate_codes(huffman_tree)
codeword_lengths = {symbol: len(code) for symbol, code in codebook.items()}

# Step 6: Plotting (only for non-zero probability symbols)
symbols = sorted(codeword_lengths.keys())
lengths = [codeword_lengths[s] for s in symbols]

plt.figure(figsize=(14, 6))
plt.bar(symbols, lengths, width=1.0)
plt.title("Huffman Codeword Lengths for Residuals")
plt.xlabel("Residual Error Value")
plt.ylabel("Codeword Length (bits)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Print stats
print("Number of codewords:", len(codebook))
print("Max codeword length:", max(lengths))
print("Min codeword length:", min(lengths))

# Step 8: Check prefix property
def is_prefix_free(codebook):
    codes = list(codebook.values())
    for i, code1 in enumerate(codes):
        for j, code2 in enumerate(codes):
            if i != j and code2.startswith(code1):
                return False
    return True

print("Prefix-free:", is_prefix_free(codebook))
