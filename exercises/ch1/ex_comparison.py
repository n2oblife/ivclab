import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread
from ivclab.utils.metrics import calc_psnr

from exE import codec, codec_postfiltering, subsampling, subsampling_postfiltering
from ex_ict import codec_ict

# ----------------------------
# Helper to compute bitrate and PSNR
# ----------------------------
def compute_metrics(original, compressed, bits_total):
    H, W = original.shape[:2]
    bpp = bits_total / (H * W)
    psnr = calc_psnr(original.astype(np.float32), compressed.astype(np.float32), maxval=255)
    return bpp, psnr

# ----------------------------
# Run all compression schemes
# ----------------------------
def evaluate_methods(image_path):
    image = imread(image_path)    
    results = {}
    low_pass_kernel = np.asarray(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=float
    )

    # 1. codec (lowpass + down+up sample)
    _, _, up_img = codec(image, low_pass_kernel)
    bpp = 3 * 8 * 0.25  # downsampled by factor 2 in each dim: 1/4 of original
    results["codec"] = (bpp, calc_psnr(image, up_img, maxval=255))

    # 2. codec_postfiltering (lowpass -> down+up -> postfilter)
    _, _, _, postfiltered = codec_postfiltering(image, low_pass_kernel)
    results["codec_postfiltering"] = (bpp, calc_psnr(image, postfiltered, maxval=255))

    # 3. subsampling (without prefiltering)
    _, subsampled = subsampling(image)
    results["subsampling"] = (bpp, calc_psnr(image, subsampled, maxval=255))

    # 4. subsampling_postfiltering
    _, _, sub_postfiltered = subsampling_postfiltering(image, low_pass_kernel)
    results["subsampling_postfiltering"] = (bpp, calc_psnr(image, sub_postfiltered, maxval=255))

    # 5. codec_ict (ICT-based chroma subsampling)
    compressed = codec_ict(image)
    bpp_ict = 8 * (1 + 2 * 0.25)  # Y full, Cb/Cr subsampled: 8 * 1.5
    results["codec_ict"] = (bpp_ict, calc_psnr(image, compressed, maxval=255))

    return results

# ----------------------------
# Visualize D-R curves
# ----------------------------
def plot_rate_distortion(results_dict, image_names):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', '*']
    color_map = {
        "codec": "blue",
        "codec_postfiltering": "green",
        "subsampling": "red",
        "subsampling_postfiltering": "orange",
        "codec_ict": "purple"
    }

    for i, method in enumerate(results_dict[next(iter(results_dict))].keys()):
        for j, img in enumerate(image_names):
            bpp, psnr = results_dict[img][method]
            color = color_map.get(method, 'black')
            marker = markers[i % len(markers)]
            plt.scatter(bpp, psnr, color=color, marker=marker, label=method if j == 0 else "")
            plt.text(bpp + 0.02, psnr, img.split('.')[0], fontsize=8)  # offset to avoid overlap

    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Performance of Compression Techniques")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_rate_distortion(results_dict):
    import numpy as np
    plt.figure(figsize=(8, 6))
    
    color_map = {
        "codec": "blue",
        "codec_postfiltering": "green",
        "subsampling": "red",
        "subsampling_postfiltering": "orange",
        "codec_ict": "purple"
    }
    
    methods = results_dict[next(iter(results_dict))].keys()
    
    for method in methods:
        bpp_list = []
        psnr_list = []
        for img_results in results_dict.values():
            bpp, psnr = img_results[method]
            bpp_list.append(bpp)
            psnr_list.append(psnr)
        
        avg_bpp = np.mean(bpp_list)
        avg_psnr = np.mean(psnr_list)
        
        plt.scatter(avg_bpp, avg_psnr, color=color_map.get(method, 'black'), label=method)
        plt.text(avg_bpp + 0.01, avg_psnr, method, fontsize=9)

    plt.xlabel("Average Bits per pixel (bpp)")
    plt.ylabel("Average PSNR (dB)")
    plt.title("Average Rate-Distortion per Method")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


# ----------------------------
# Main runner
# ----------------------------
if __name__ == "__main__":
    image_files = [
        "data/lena.tif",
        "data/monarch.tif",
        "data/sail.tif",
        "data/smandril.tif",
        "data/peppers.tif"
    ]

    image_names = [name.split('/')[-1].split('.')[0] for name in image_files]
    all_results = {}

    for img_path in image_files:
        print(f"Processing {img_path} ...")
        result = evaluate_methods(img_path)
        all_results[img_path.split('/')[-1].split('.')[0]] = result

    plot_rate_distortion(all_results, image_names)
    plot_average_rate_distortion(all_results)

