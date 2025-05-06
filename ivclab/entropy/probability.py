import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view
from ivclab.utils import imread
from ivclab.signal import rgb2gray

import numpy as np

# def basic_histo(image: np.ndarray) -> np.ndarray:
#     """Compute a basic histogram of intensity values for 8-bit images."""
#     image = image.flatten()  # Flatten to 1D array
#     image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure valid uint8 range
#     histo = np.zeros(256, dtype=int)
#     for px in image:
#         histo[px] += 1
#     return histo

def basic_histo(image: np.ndarray):
    """
    Compute histogram(s) of intensity values for 8-bit grayscale or RGB images.

    Returns:
        - Grayscale: (256,) histogram
        - RGB: tuple of three (256,) histograms for R, G, B
    """
    if image.ndim == 2:  # Grayscale image
        flat = image.flatten()
        flat = np.clip(flat, 0, 255).astype(np.uint8)
        histo = np.zeros(256, dtype=int)
        for px in flat:
            histo[px] += 1
        return histo

    elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
        histos = []
        for channel in range(3):
            flat = image[:, :, channel].flatten()
            flat = np.clip(flat, 0, 255).astype(np.uint8)
            hist = np.zeros(256, dtype=int)
            for px in flat:
                hist[px] += 1
            histos.append(hist)
        return tuple(histos)

    else:
        raise ValueError("Unsupported image format. Must be 2D grayscale or 3D RGB.")


def count_rgb_histogram(image_path, grayscale=False):
    img = imread(image_path)
    if grayscale:
        img = rgb2gray(img)

    if img.ndim == 2:  # grayscale
        histogram = np.bincount(img.flatten(), minlength=256)
        print(f"Grayscale image: 256 bins used.")
        return histogram
    
    # Color image: count 3D bins
    flat_pixels = img.reshape(-1, img.shape[2])
    
    # Convert RGB triplets to a single integer value (base-256 encoding)
    rgb_flat = flat_pixels[:, 0] * 256**2 + flat_pixels[:, 1] * 256 + flat_pixels[:, 2]
    histogram = Counter(rgb_flat)

    print(f"Color image: {len(histogram)} bins used out of 16,777,216 possible.")
    return histogram

def plot_histogram(image_path, grayscale=False):
    img = imread(image_path)
    if grayscale:
        img = rgb2gray(img)
        # Convert float grayscale [0,1] to uint8 [0,255]
        if img.dtype != np.uint8:
            img = (img).astype(np.uint8)
        else:
            img = img

    plt.figure(figsize=(18, 4))  # wider for image + histograms
    plt.suptitle(f"Histogram for {image_path.split('/')[-1]}")

    # Show original image on the left
    plt.subplot(1, 2 if grayscale else 4, 1)
    plt.imshow(img, cmap='gray' if grayscale else None)
    plt.axis('off')
    plt.title("Original Image")

    if grayscale:
        hist = np.bincount(img.flatten(), minlength=256)
        plt.subplot(1, 2, 2)
        plt.bar(range(256), hist, color='gray')
        plt.title("Grayscale Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    else:  # RGB image
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.subplot(1, 4, i + 2)
            hist = np.bincount(img[:, :, i].flatten(), minlength=256)
            plt.bar(range(256), hist, color=color)
            plt.title(f"{color.upper()} Channel")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_image_and_joint_histogram(image, joint_pmf, title, to_gray=False):
    # Reshape 1D joint PMF into 2D matrix (256 x 256)
    # joint_matrix = joint_pmf.reshape((256, 256))
    joint_matrix = joint_pmf

    # Create a figure with two subplots
    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)
    if to_gray:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.title(f"Original Image: {title}")
    plt.axis('off')

    # Joint histogram
    plt.subplot(1, 2, 2)
    plt.imshow(joint_matrix, cmap='hot', interpolation='nearest')
    plt.title("Joint Histogram (horizontal pairs)")
    plt.xlabel("Pixel i")
    plt.ylabel("Pixel i+1")
    plt.colorbar(label="Probability")

    plt.tight_layout()
    plt.show()

def stats_joint(image, pixel_range, to_flat=False):
    """
    Computes joint probability of non-overlapping horizontal pixel pairs
    of an image, similar to stats_marg function. However, this
    counts every instance of pixel pairs in a 2D table to
    find the frequencies. Then, it normalizes the values to
    convert them to probabilities. Return a 1D vector
    since this is how we represent pmf values.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        pmf: np.array of shape [B^2], probability mass function of image pixel pairs over range
    """
    # A table to hold count of pixel pair occurences
    count_table = np.zeros((len(pixel_range), len(pixel_range)))

    # Get all non overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(image, 'h (w s) c -> (h w c) s', s=2)

    # YOUR CODE STARTS HERE
    hist2d, _, _ = np.histogram2d(
        pixel_pairs[:, 0],  # First pixel in pair
        pixel_pairs[:, 1],  # Second pixel in pair
        bins=[pixel_range, pixel_range]
    )

    count_table = hist2d / np.sum(hist2d)
    pmf = count_table.flatten() if to_flat else count_table
    # YOUR CODE ENDS HERE
    return pmf

def stats_cond(image, pixel_range, eps=1e-8, to_flat=False):
    """
    Computes conditional probability of overlapping horizontal pixel pairs
    of an image, similar to stats_joint function. The conditional probability
    is found by the formula SUM{ - p(x,y) * ( log2( p(x,y) ) - log2( p(x) ) ) }. To compute
    p(x), you can take the sum of normalized probabilities of p(x,y) over row axis.
    Make sure to add a small epsilon before computing the log probabilities. You can
    ignore the first pixels in every row since they don't have a left neighbor.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        cond_entropy: a scalar value
    """
    # A table to hold count of pixel pair occurences
    pmf_table = np.zeros((len(pixel_range), len(pixel_range)))

    # Get all overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(sliding_window_view(image, 2, axis=1), 'h w c s-> (h w c) s', s=2) 

    # YOUR CODE STARTS HERE
    hist2d, _, _ = np.histogram2d(
        pixel_pairs[:, 0],  # First pixel in pair
        pixel_pairs[:, 1],  # Second pixel in pair
        bins=[pixel_range, pixel_range]
    )

    pmf_table = hist2d / np.sum(hist2d)
    pmf = pmf_table.flatten() if to_flat else pmf_table
    p_x = pmf_table.sum(axis=1)    

    # Add small epsilon to avoid log(0)
    pmf += eps
    p_x += eps

    cond_entropy = -np.sum(pmf * (np.log2(pmf) - np.log2(p_x[:, np.newaxis])))
    # YOUR CODE ENDS HERE
    return cond_entropy

if __name__ == "__main__":
    image_paths = ["data/lena.tif", "data/sail.tif", "data/peppers.tif"]
    pmfs = {}
    for path in image_paths:
        image = imread(path)
        image = rgb2gray(image)
        pmfs[path] = stats_joint(image, np.arange(256))
        plot_image_and_joint_histogram(image, pmfs[path], path)
        
        # plot_histogram(path, grayscale=False)
        # plot_histogram(path, grayscale=True)
