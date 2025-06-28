import numpy as np
from einops import rearrange

class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size = 64):
        self.EOB = end_of_block
        self.block_size = block_size

    def encode(self, flat_patch_img):
        """
        Encode each (64,) block independently to avoid stream desyncs.
        """
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p')
        encoded = []
        for block in flat_img:
            last_nonzero = self.block_size - 1
            while last_nonzero >= 0 and block[last_nonzero] == 0:
                last_nonzero -= 1

            if last_nonzero == -1:
                encoded.append(self.EOB)
                continue
            i = 0
            while i <= last_nonzero:
                val = block[i]
                if val == 0:
                    run_len = 1
                    while (i + run_len <= last_nonzero) and (block[i + run_len] == 0):
                        run_len += 1
                    encoded.extend([0, run_len])
                    i += run_len
                else:
                    encoded.append(int(val))
                    i += 1
            encoded.append(self.EOB)
        return np.array(encoded, dtype=np.int32)

    
    def decode(self, encoded, original_shape):
        """
        This function gets an encoding and the original shape to decode the elements 
        of the original array. It acts as the inverse function of the encoder.

        encoded: List of symbols that represent the original elements
        original_shape: List of 3 numbers that represent number of H_patch, W_patch and C

        returns:
            flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]
        
        """
        h, w, c = original_shape
        flat_img = []
        i = 0
        expected_blocks = h * w * c
        while i < len(encoded) and len(flat_img) < expected_blocks:
            block = []
            while True:
                if i >= len(encoded):
                    raise ValueError("Unexpected end of encoded symbols")
                symbol = encoded[i]
                i += 1
                if symbol == self.EOB:
                    block.extend([0] * (self.block_size - len(block)))
                    break
                elif symbol == 0:
                    run_len = encoded[i]
                    i += 1
                    block.extend([0] * run_len)
                else:
                    block.append(symbol)
                if len(block) > self.block_size:
                    raise ValueError(f"Block size exceeded: {len(block)}")
            if len(block) != self.block_size:
                raise ValueError(f"Incomplete block: {len(block)}")
            flat_img.append(block)
        if len(flat_img) != expected_blocks:
            raise ValueError(f"Expected {expected_blocks} blocks, got {len(flat_img)}")
        flat_img = np.array(flat_img, dtype=np.int32)
        return rearrange(flat_img, '(h w c) p -> h w c p', h=h, w=w, c=c, p=self.block_size)    