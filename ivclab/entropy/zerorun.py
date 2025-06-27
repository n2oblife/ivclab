import numpy as np
from einops import rearrange

class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size = 64):
        self.EOB = end_of_block
        self.block_size = block_size

    def encode(self, flat_patch_img: np.array):
        """
        This function gets a flattened patched image and produces a list of 
        symbols that applies a zero run encoding of the input where sequential
        blocks of zeroes (e.g. [... 0 0 0 0 0 ...]) are replaced with a marker zero
        and the number of additional zeroes (e.g. [... 0 4 ...]). The original sequence
        is processed in blocks of block_size and every encoding of a block ends with an
        end of block symbol. If all the original values are zero until the end of block,
        then no marker is necessary and we can put an EOB symbol directly.

        flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]

        returns:
            encoded: List of symbols that represent the original elements
        
        """
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p', p=self.block_size)
        encoded = []

        for block in flat_img:
            i = 0
            # Find last non-zero element to determine cutoff
            last_nonzero = self.block_size - 1
            while last_nonzero >= 0 and block[last_nonzero] == 0:
                last_nonzero -= 1

            if last_nonzero == -1:
                # All zeros: just insert EOB
                encoded.append(self.EOB)
                continue

            while i <= last_nonzero:
                if block[i] == 0:
                    # Count run of zeros
                    run_len = 1
                    while (i + run_len <= last_nonzero) and (block[i + run_len] == 0):
                        run_len += 1
                    encoded.extend([0, run_len])
                    i += run_len
                else:
                    encoded.append(int(block[i]))
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
            while len(block) < self.block_size:
                if i >= len(encoded):
                    raise ValueError("Unexpected end of encoded symbols during block reconstruction.")

                symbol = encoded[i]
                if symbol == self.EOB:
                    block.extend([0] * (self.block_size - len(block)))
                    i += 1
                    break
                elif symbol == 0:
                    if i + 1 >= len(encoded):
                        raise ValueError("Run-length zero symbol without a following count.")
                    run_len = encoded[i + 1]
                    if len(block) + run_len > self.block_size:
                        raise ValueError(f"Zero run overflows block: block size {self.block_size}, run_len {run_len}")
                    block.extend([0] * run_len)
                    i += 2
                else:
                    block.append(symbol)
                    i += 1

            # Check block size again
            if len(block) != self.block_size:
                raise ValueError(f"Decoded block has invalid length: {len(block)} (expected {self.block_size})")

            # Only append if we've collected a full block
            flat_img.append(block)

        flat_img = np.array(flat_img, dtype=np.int32)
        # sanity check
        expected_blocks = h * w * c
        actual_blocks = flat_img.shape[0]

        if expected_blocks != actual_blocks:
            raise ValueError(f"Mismatch in number of blocks: expected {expected_blocks}, got {actual_blocks}")

        # if flat_img.shape[0] > expected_blocks:
        #     flat_img = flat_img[:expected_blocks]
        # elif flat_img.shape[0] < expected_blocks:
        #     raise ValueError("Too few blocks recovered in ZeroRun decoding")

        flat_patch_img = rearrange(flat_img, '(h w c) p -> h w c p',
                                h=h, w=w, c=c, p=self.block_size)

        return flat_patch_img

            
            