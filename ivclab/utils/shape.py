import numpy as np
from einops import rearrange

class ZigZag:
    """
    An object that flattens two dimensional patches according to a zigzag rule
    on a 8x8 grid
    """
    def __init__(self):
        self.zigzag_order = np.asarray([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])

    def flatten(self, patched_img: np.array):
        print(f"🔍 DEBUG: ZigZag.flatten - Input shape: {patched_img.shape}")
        flattened = rearrange(patched_img, 'h w c p0 p1 -> h w c (p0 p1)')
        print(f"🔍 DEBUG: ZigZag.flatten - After rearrange: {flattened.shape}")
        shuffled = np.zeros_like(flattened)
        shuffled[:,:,:,self.zigzag_order] = flattened
        print(f"🔍 DEBUG: ZigZag.flatten - After zigzag shuffle: {shuffled.shape}")
        return shuffled

    def unflatten(self, unshuffled):
        print(f"🔍 DEBUG: ZigZag.unflatten - Input shape: {unshuffled.shape}")
        shuffled = unshuffled[:,:,:,self.zigzag_order]
        print(f"🔍 DEBUG: ZigZag.unflatten - After zigzag unshuffle: {shuffled.shape}")
        unflattened = rearrange(shuffled, 'h w c (p0 p1) -> h w c p0 p1', p0=8, p1=8)
        print(f"🔍 DEBUG: ZigZag.unflatten - After rearrange: {unflattened.shape}")
        return unflattened

class Patcher:
    """
    A class to extract/merge patches from/to an image 
    """
    def __init__(self, window_size=(8,8)):
        self.window_size = window_size

    def patch(self, img: np.array):
        """
        Extracts patches from an image.

        img: np.array of shape [H, W, C]

        returns:
            patched_img = [H_patch, W_patch, C, H_window, W_window]
        """
        return rearrange(img, '(h p0) (w p1) c -> h w c p0 p1', p0=self.window_size[0], p1=self.window_size[1])
    
    def unpatch(self, patched_img: np.array):
        """
        Merges patches to an image.

        patched_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]

        returns:
            img = [H, W, C]
        """
        return rearrange(patched_img, 'h w c p0 p1 -> (h p0) (w p1) c', p0=self.window_size[0], p1=self.window_size[1])
    