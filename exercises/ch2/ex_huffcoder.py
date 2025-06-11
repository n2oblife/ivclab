import numpy as np
import matplotlib.pyplot as plt
import constriction
from PIL import Image
from collections import Counter
import os

class HuffmanCoder:
    def __init__(self, lower_bound=0):
        self.lower_bound = lower_bound
        self.probs = None
        self.encoder_codebook = None
        self.decoder_codebook = None
        self.codewords = {}
        self.codelengths = {}
    
    def train(self, probs):
        """Train the Huffman coder with probability mass function"""
        self.probs = probs
        self.encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probs)
        self.decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probs)
        
        # Extract codewords and lengths
        self.codewords = {}
        self.codelengths = {}
        for i in range(len(probs)):
            code = self.encoder_codebook.get_code(i)
            code_str = ''.join(str(b) for b in code)
            self.codewords[i + self.lower_bound] = code_str
            self.codelengths[i + self.lower_bound] = len(code_str)
    
    def encode(self, message):
        """Encode a message using the trained Huffman code"""
        if self.encoder_codebook is None:
            raise ValueError("Huffman codec must be trained first with the probabilities: call '.train(probs)'")
        
        encoder = constriction.symbol.QueueEncoder()
        for symbol in message:
            encoder.encode_symbol(symbol - self.lower_bound, self.encoder_codebook)
        
        compressed, bitrate = encoder.get_compressed()
        return np.asarray(compressed), bitrate
    
    def decode(self, compressed, message_length):
        """Decode a compressed message"""
        if self.decoder_codebook is None:
            raise ValueError("Huffman codec must be trained first with the probabilities: call '.train(probs)'")
        
        decoder = constriction.symbol.QueueDecoder(compressed)
        decoded = []
        for i in range(message_length):
            symbol = decoder.decode_symbol(self.decoder_codebook)
            decoded.append(symbol + self.lower_bound)
        
        return np.asarray(decoded)
    
    def is_prefix_free(self):
        """Check if the code is prefix-free"""
        code_strs = list(self.codewords.values())
        for i in range(len(code_strs)):
            for j in range(len(code_strs)):
                if i != j and code_strs[i].startswith(code_strs[j]):
                    return False
        return True
    
    def get_individual_compression_lengths(self, symbols):
        """Get compression length for each individual symbol"""
        lengths = {}
        for symbol in symbols:
            if symbol in self.codelengths:
                lengths[symbol] = self.codelengths[symbol]
            else:
                lengths[symbol] = 0  # Symbol not in codebook
        return lengths

def minimum_entropy_predictor(image):
    """
    Implement minimum-entropy predictor for image compression.
    For each pixel, predict based on neighbors and compute residual.
    """
    height, width = image.shape
    predicted = np.zeros_like(image)
    residuals = []
    
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                # First pixel: no prediction possible
                prediction = 128  # Use middle gray value
            elif i == 0:
                # First row: predict from left neighbor
                prediction = image[i, j-1]
            elif j == 0:
                # First column: predict from upper neighbor
                prediction = image[i-1, j]
            else:
                # General case: use minimum entropy predictor
                # Consider three neighbors: N (north), W (west), NW (northwest)
                N = image[i-1, j]
                W = image[i, j-1]
                NW = image[i-1, j-1]
                
                # Minimum entropy predictor: min(N, W) if NW >= max(N, W)
                # max(N, W) if NW <= min(N, W), otherwise N + W - NW
                if NW >= max(N, W):
                    prediction = min(N, W)
                elif NW <= min(N, W):
                    prediction = max(N, W)
                else:
                    prediction = N + W - NW
            
            predicted[i, j] = prediction
            residual = int(image[i, j]) - int(prediction)
            residuals.append(residual)
    
    return np.array(residuals), predicted

def load_or_create_test_image(filename='lena_small.tif'):
    """Load the specified image or create a test image if not available"""
    try:
        # Try to load the specified image
        img = Image.open(filename)
        if img.mode != 'L':
            img = img.convert('L')  # Convert to grayscale
        return np.array(img)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Creating a synthetic test image.")
        # Create a synthetic test image with some structure
        size = 64
        x, y = np.meshgrid(np.linspace(0, 4*np.pi, size), np.linspace(0, 4*np.pi, size))
        image = ((np.sin(x) * np.cos(y) + 1) * 127.5).astype(np.uint8)
        # Add some noise
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return image

def analyze_huffman_coding():
    """Main function to analyze Huffman coding of predictor residuals"""
    
    # Load image
    print("Loading image...")
    image = load_or_create_test_image('lena_small.tif')
    print(f"Image shape: {image.shape}")
    
    # Apply minimum entropy predictor
    print("Applying minimum entropy predictor...")
    residuals, predicted = minimum_entropy_predictor(image)
    
    print(f"Residual range: [{residuals.min()}, {residuals.max()}]")
    print(f"Number of residuals: {len(residuals)}")
    
    # Calculate residual statistics
    residual_counts = Counter(residuals)
    all_possible_residuals = range(residuals.min(), residuals.max() + 1)
    
    # Create probability mass function covering all possible residuals
    total_count = len(residuals)
    residual_probs = []
    residual_symbols = []
    
    for residual in all_possible_residuals:
        count = residual_counts.get(residual, 0)
        # Add small probability for symbols not in test set to ensure coverage
        prob = max(count / total_count, 1e-6)
        residual_probs.append(prob)
        residual_symbols.append(residual)
    
    # Normalize probabilities
    residual_probs = np.array(residual_probs, dtype=np.float32)
    residual_probs = residual_probs / residual_probs.sum()
    
    print(f"Number of possible residual symbols: {len(residual_symbols)}")
    print(f"Probability mass function sum: {residual_probs.sum():.6f}")
    
    # Create and train Huffman coder
    print("Training Huffman coder...")
    huffman = HuffmanCoder(lower_bound=residuals.min())
    huffman.train(residual_probs)
    
    # Verify prefix-free property
    is_prefix_free = huffman.is_prefix_free()
    print(f"Code is prefix-free: {is_prefix_free}")
    
    # Display some codewords
    print("\nSample codewords:")
    sample_symbols = sorted(residual_symbols)[::max(1, len(residual_symbols)//10)]
    for symbol in sample_symbols[:10]:
        if symbol in huffman.codewords:
            print(f"Symbol {symbol:3d}: '{huffman.codewords[symbol]}' (length: {huffman.codelengths[symbol]})")
    
    # Calculate compression lengths for individual symbols
    individual_lengths = huffman.get_individual_compression_lengths(residual_symbols)
    
    # Compress the entire residual sequence
    print("\nCompressing residual sequence...")
    compressed, bitrate = huffman.encode(residuals)
    print(f"Original length: {len(residuals)} symbols")
    print(f"Compressed length: {len(compressed)} words ({bitrate} bits)")
    print(f"Compression ratio: {len(residuals) * 8 / bitrate:.2f}:1")
    
    # Verify decompression
    decoded = huffman.decode(compressed, len(residuals))
    decompression_successful = np.array_equal(residuals, decoded)
    print(f"Decompression successful: {decompression_successful}")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot 2: Residual histogram
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Code lengths vs symbols
    symbols_sorted = sorted(residual_symbols)
    lengths_sorted = [huffman.codelengths.get(s, 0) for s in symbols_sorted]
    
    ax3.plot(symbols_sorted, lengths_sorted, 'b.-', markersize=3)
    ax3.set_title('Huffman Code Lengths vs Residual Values')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Code Length (bits)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Prefix-free verification visualization
    # Show code length distribution
    length_counts = Counter(huffman.codelengths.values())
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    
    ax4.bar(lengths, counts, alpha=0.7, edgecolor='black')
    ax4.set_title(f'Code Length Distribution\n(Prefix-free: {is_prefix_free})')
    ax4.set_xlabel('Code Length (bits)')
    ax4.set_ylabel('Number of Symbols')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print compression statistics
    print(f"\nCompression Statistics:")
    print(f"Average code length: {np.mean(lengths_sorted):.2f} bits")
    print(f"Theoretical entropy: {-np.sum(residual_probs * np.log2(residual_probs + 1e-10)):.2f} bits")
    print(f"Huffman efficiency: {(-np.sum(residual_probs * np.log2(residual_probs + 1e-10)) / np.mean(lengths_sorted) * 100):.1f}%")
    
    return huffman, residuals, residual_symbols, individual_lengths

if __name__ == '__main__':
    # Run the complete analysis
    huffman_coder, residuals, symbols, individual_lengths = analyze_huffman_coding()
    
    # Display individual symbol compression lengths
    print(f"\nIndividual symbol compression lengths (first 20):")
    for i, (symbol, length) in enumerate(list(individual_lengths.items())[:20]):
        print(f"Symbol {symbol:3d}: {length} bits")
    
    print(f"\nTotal symbols with non-zero probability: {len([l for l in individual_lengths.values() if l > 0])}")
    print(f"Analysis complete!")