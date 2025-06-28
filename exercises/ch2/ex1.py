from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy

import numpy as np

# For this exercise, you need to implement 
# stats_marg and calc_entropy functions in
# ivclab.entropy.entropy file. You can run
# ch2 tests to make sure they are 
# correctly and you get sensible results

image_names = ['lena.tif', 'sail.tif', 'smandril.tif']

# read images
for image_name in image_names:
    img = imread(f'data/{image_name}')
    pmf_img = stats_marg(img, np.arange(256))
    entropy_img = calc_entropy(pmf_img)

    print(f"Entropy of {image_name}: H={entropy_img:.2f} bits/pixel")

