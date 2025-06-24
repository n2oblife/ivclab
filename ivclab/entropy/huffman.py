import numpy as np
import constriction
from itertools import combinations

class HuffmanCoder:
    def __init__(self, lower_bound=0):
        self.lower_bound = lower_bound
        self.probs = None
        self.encoder_codebook = None
        self.decoder_codebook = None

    def train(self, probs):
        if np.any(probs == 0):
            raise ValueError("Zero-probability symbols found in PMF. All symbols must have non-zero probability.")

        self.probs = probs
        self.encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probs)
        self.decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probs)

    def encode(self, message):
        if self.encoder_codebook is None:
            raise RuntimeError("Train the Huffman coder before encoding.")

        max_symbol = len(self.probs) - 1 + self.lower_bound
        if np.any((message < self.lower_bound) | (message > max_symbol)):
            raise ValueError("Message contains symbols outside the trained range.")

        encoder = constriction.symbol.QueueEncoder()
        for symbol in message:
            encoder.encode_symbol(symbol - self.lower_bound, self.encoder_codebook)

        compressed, bitrate = encoder.get_compressed()
        return np.asarray(compressed), float(bitrate)

    def decode(self, compressed, message_length):
        if self.decoder_codebook is None:
            raise RuntimeError("Train the Huffman coder before decoding.")

        decoder = constriction.symbol.QueueDecoder(compressed)
        decoded = [
            decoder.decode_symbol(self.decoder_codebook) + self.lower_bound
            for _ in range(message_length)
        ]
        return np.asarray(decoded)

    def is_prefix_free(self):
        codes = [self.encoder_codebook.get_code(i) for i in range(len(self.probs))]
        code_strs = [''.join(str(bit) for bit in code) for code in codes]
        for a, b in combinations(code_strs, 2):
            if a.startswith(b) or b.startswith(a):
                return False
        return True

if __name__ == '__main__':
    huffman = HuffmanCoder(lower_bound=0)
    probs = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    message = np.array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 2, 0])

    huffman.train(probs)
    compressed, bitrate = huffman.encode(message)
    print(f"Compressed: {compressed}")
    print(f"Bitrate: {bitrate:.4f} bits/symbol")

    decoded = huffman.decode(compressed, len(message))
    print("Decoding successful:", np.array_equal(decoded, message))
