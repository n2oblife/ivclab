from ivclab.utils import imread
from ivclab.entropy import stats_joint, calc_entropy
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

# For this exercise, you need to implement 
# stats_joint function in ivclab.entropy.probability file.
# You can run ch2 tests to make sure they are implemented correctly
image_names = ['lena.tif', 'sail.tif', 'smandril.tif']

for image_name in image_names:
    lena_img = imread(f'data/'+image_name)
    pixel_range = np.arange(256)
    # joint_pmf = stats_joint(lena_img, pixel_range, to_flat=True)
    joint_pmf = stats_joint(lena_img, pixel_range)
    joint_entropy = calc_entropy(joint_pmf)

    print(f"Joint entropy of {image_name}: H={joint_entropy:.2f} bits/pixel pair")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X, Y = np.meshgrid(pixel_range, pixel_range)

    # Plot the surface.
    # surf = ax.plot_surface(X, Y, rearrange(joint_pmf, '(h w) -> h w', h=256, w=256),
    #                        cmap='plasma', linewidth=0, antialiased=False)
    surf = ax.plot_surface(X, Y, joint_pmf,
                        cmap='plasma', linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f'Joint PMF of {image_name} Image')
    plt.xlabel(f'Pixel i Value')
    plt.ylabel(f'Pixel i+1 Value')
    plt.legend()
    plt.show()



