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
        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE

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

        image = np.zeros_like(ref_image)

        # YOUR CODE STARTS HERE
        # raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return image