from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy
from ivclab.entropy import stats_marg, min_code_length

import numpy as np

# For this exercise, you need to implement 
# stats_marg and calc_entropy functions in
# ivclab.entropy.entropy file. You can run
# ch2 tests to make sure they are implemented
# correctly and you get sensible results

image_names = ['lena.tif', 'sail.tif', 'smandril.tif']

all_entropy = {}

# read images
for image_name in image_names:
    img = imread(f'data/{image_name}')
    pmf_img = stats_marg(img, np.arange(256))
    entropy_img = calc_entropy(pmf_img)
    all_entropy[image_name] = entropy_img

    print(f"Entropy of {image_name}: H={entropy_img:.2f} bits/pixel")

all_pmfs = {}

# read images to compute pmfs and common_pmf
for image_name in image_names:
    img = imread(f'data/{image_name}')
    pmf_img = stats_marg(img, np.arange(256))
    all_pmfs[image_name] = pmf_img

# common_pmf = (all_pmfs[0] + all_pmfs[1] + all_pmfs[2]) / 3
common_pmf = np.mean([pmf for pmf in all_pmfs.values()])
all_code_length = {}

for image_name, target_pmf in zip(image_names, all_pmfs):
    code_length = min_code_length(target_pmf, common_pmf)
    all_code_length[image_name] = code_length
    print(f"Minimum average codeword length of {image_name} under common table: H={code_length:.2f} bits/pixel")

all_diffs = {}
print(f"The difference we are computing is the value : code_length - entropy")
for image_name, code_length, entropy in zip(image_names, all_code_length, all_entropy):
    diff = code_length - entropy
    all_diffs[image_name] = diff
    print(f"Difference between combined table and entropy of {image_name}: dH={diff:.2f} bits/pixel")
