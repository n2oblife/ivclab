import numpy as np
from ivclab.utils import imread
from ivclab.signal import rgb2gray
from ivclab.entropy.probability import stats_cond

def stats_marg(image, pixel_range):
    """
    Computes marginal probability of the pixels of an image
    by counting them with np.histogram and normalizing 
    using the total count. Do not forget to flatten the image
    first. You can pass the range as 'bins' argument to the 
    histogram function. The histogram function returns the counts
    and the bin edges.

    image: np.array of any shape, preferably [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        pmf: np.array of shape [B], probability mass function of image pixels over range
    """
    # Convert to float for safety (optional, since np.histogram works fine with uint8 too)
    image = image.astype(np.float64)

    # YOUR CODE STARTS HERE
    flattened = image.flatten()
    counts, _ = np.histogram(flattened, bins=pixel_range)

    total_pixels = flattened.size # Normalize
    pmf = counts / total_pixels
    # YOUR CODE ENDS HERE
    return pmf

def calc_entropy(pmf, eps=1e-8):
    """
    Computes entropy for the given probability mass function
    with the formula SUM{ - p(x) * log2(p(x))}.

    pmf: np.array of shape [B] containing the probabilities for bins

    returns 
        entropy: scalar value, computed entropy according to the above formula
    """
    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    # pmf = pmf + eps
    nonzero_pmf = pmf[pmf > 0]
    
    # YOUR CODE STARTS HERE
    entropy = -np.sum(nonzero_pmf * np.log2(nonzero_pmf))
    # YOUR CODE ENDS HERE
    return entropy

def min_code_length(target_pmf, common_pmf, eps=1e-8):
    """
    Computes minimum average codeword length for the
    target pmf given the common pmf using the formula
    formula SUM{ - p(x) * log2(q(x))} where p(x) is the
    target probability and q(x) comes from the common pmf

    target_pmf: np.array of shape [B] containing the probabilities for bins
    common_pmf: np.array of shape [B] containing the probabilities for bins

    returns 
        code_length: scalar value, computed entropy according to the above formula
    """
    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    common_pmf = common_pmf + eps
    # non_zero_common_pmf = common_pmf[common_pmf > 0]
    
    # YOUR CODE STARTS HERE
    code_length = -np.sum(target_pmf * np.log2(common_pmf))
    # YOUR CODE ENDS HERE
    return code_length

if __name__ == "__main__":
    image_paths = ["data/lena.tif", "data/sail.tif", "data/peppers.tif"]

    pmfs = {}
    entropies = {}
    code_lengths = {}
    diffs = {}
    cond_entropies = {}
    cond_diffs = {}

    # Calculate PMFs and entropies for individual images
    for path in image_paths:
        image = imread(path)
        image_gray = rgb2gray(image)

        pmf = stats_marg(image_gray, np.arange(256))  # bins = 256, range = [0, 256]
        pmfs[path] = pmf

        entropy = calc_entropy(pmf)
        entropies[path] = entropy
        print(f"Entropy for {path}: {entropy:.4f} bits")

    # Average entropy
    average_entropy = sum(entropies.values()) / len(entropies)
    print(f"Average entropy for all images: {average_entropy:.4f} bits\n")

    # Compute common PMF
    common_pmf = np.mean(np.array(list(pmfs.values())), axis=0)

    # Compute code lengths using common PMF
    for path in image_paths:
        code_length = min_code_length(target_pmf=pmfs[path], common_pmf=common_pmf)
        code_lengths[path] = code_length
        print(f"Minimum codeword length for {path} (using common PMF): {code_length:.4f} bits")

    # Average codeword length
    average_code_length = sum(code_lengths.values()) / len(code_lengths)
    print(f"Average codeword length using common PMF: {average_code_length:.4f} bits\n")

    # Compute and display differences
    for path in image_paths:
        diff = code_lengths[path] - entropies[path]
        diffs[path] = diff
        print(f"Difference for {path}: {diff:.4f} bits")

    # Average difference
    average_diff = sum(diffs.values()) / len(diffs)
    print(f"Average difference (code length - entropy): {average_diff:.4f} bits\n")

    # Conditional entropy
    for path in image_paths:
        image = imread(path)
        image_gray = rgb2gray(image)

        cond_entropy = stats_cond(image_gray, np.arange(256))
        cond_entropies[path] = cond_entropy

        print(f"Conditional entropy for {path}: {cond_entropies[path]:.4f} bits")