from ivclab.utils import imread
from ivclab.entropy import stats_marg, min_code_length

import numpy as np

# For this exercise, you need to implement min_code_length
# function in ivclab.entropy.entropy file. You can run 
# ch2 tests to make sure it is implemented correctly.

image_names = ['lena.tif', 'sail.tif', 'smandril.tif']

all_pmfs = list()

# read images to compute pmfs and common_pmf
for image_name in image_names:
    img = imread(f'data/{image_name}')
    pmf_img = stats_marg(img, np.arange(256))
    all_pmfs.append(pmf_img)

common_pmf = (all_pmfs[0] + all_pmfs[1] + all_pmfs[2]) / 3
breakpoint()

for image_name, target_pmf in zip(image_names, all_pmfs):
    code_length = min_code_length(target_pmf, common_pmf)
    print(f"Minimum average codeword length of {image_name} under common table: H={code_length:.2f} bits/pixel")

