import numpy as np

class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range

    def compute_motion_vector(self, ref_image, image):
        """
        Computes the motion vector that describes the motion between the single channeled reference image and the current image.
        The motion vector is represented as a 2D numpy array with shape [H / 8, W / 8, 1], where the x and y displacements
        are converted to a single value using the formula: motion_vector = y_displacement * (2 * search_range + 1) + x_displacement. Notice that
        displacements can take any value in the range [-search_range, search_range]. We compute the motion vectors only for
        the 8x8 non-overlapping blocks. Compute the closest indice using sum of the squared differences (SSD) between the reference block and the current block.

        ref_image: np.array of shape [H, W]
        image: np.array of shape [H, W]

        returns:
            motion_vector: np.array of shape [H / 8, W / 8, 1]
        """
        H, W = ref_image.shape
        block_size = 8
        sr = self.search_range
        mv_shape = (H // block_size, W // block_size, 1)
        motion_vector = np.zeros(mv_shape, dtype=int)

        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                block = image[i:i + block_size, j:j + block_size]
                min_ssd = float('inf')
                best_dx = 0
                best_dy = 0

                for dy in range(-sr, sr + 1):
                    for dx in range(-sr, sr + 1):
                        ref_i = i + dy
                        ref_j = j + dx

                        # Check bounds
                        if (ref_i < 0 or ref_i + block_size > H or
                                ref_j < 0 or ref_j + block_size > W):
                            continue

                        ref_block = ref_image[ref_i:ref_i + block_size, ref_j:ref_j + block_size]
                        ssd = np.sum((block - ref_block) ** 2)

                        if ssd < min_ssd:
                            min_ssd = ssd
                            best_dx = dx
                            best_dy = dy

                mv_y = i // block_size
                mv_x = j // block_size
                index = (best_dy + sr) * (2 * sr + 1) + (best_dx + sr)
                motion_vector[mv_y, mv_x, 0] = index

        return motion_vector.astype(int)

    def reconstruct_with_motion_vector(self, ref_image, motion_vector):
        """
        Reconstructs the current image using the reference image and the motion vector. The motion vector is used to
        displace the 8x8 blocks in the reference image to their corresponding positions in the current image.

        ref_image: np.array of shape [H, W, C]
        motion_vector: np.array of shape [H / 8, W / 8, 1]

        returns:
            image: np.array of shape [H, W, C]
        """
        H, W, C = ref_image.shape
        block_size = 8
        sr = self.search_range
        reconstructed = np.zeros_like(ref_image)

        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                mv_y = y // block_size
                mv_x = x // block_size
                index = motion_vector[mv_y, mv_x, 0]
                total_range = 2 * sr + 1

                dy = index // total_range - sr
                dx = index % total_range - sr

                ref_y = y + dy
                ref_x = x + dx

                # Check bounds and skip if out of image (should not happen if estimated properly)
                if (ref_y < 0 or ref_y + block_size > H or
                        ref_x < 0 or ref_x + block_size > W):
                    continue

                reconstructed[y:y + block_size, x:x + block_size, :] = \
                    ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size, :]

        return reconstructed
