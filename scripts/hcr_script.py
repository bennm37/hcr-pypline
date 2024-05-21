from hcrp.hcr import quantify_hcr
from hcrp.labelling import get_mean_region
from skimage.io import imread, imsave
import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    read_stack = True
    if read_stack:
        stack = imread(
            "data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant.tif"
        )
        image = stack[26, :, :, 1]
        # write the image
        imsave("data/seperated.tif", image)
    else:
        image = imread("data/seperated.tif")
    image = (image // 256).astype(np.uint8)
    normalized = cv2.equalizeHist(image)
    background = get_mean_region(normalized, None, "Background", size=50, vmax=None)
    print(background)
    labels, centroids = quantify_hcr(
        image,
        mean_background=0.5,
        sigma_blur=0.2,
        fg_width=0.05,
        dot_intensity_thresh=0.1,
        verbose=True,
    )
    n_centroids = centroids.shape[0]
    p = 1.0 * np.arange(n_centroids) / (n_centroids - 1)
    fig, ax = plt.subplots()
    ax.plot(np.sort(centroids[:, 0]), p, c="r")
    ax.set(xlabel="Centroid Location", ylabel="CDF")
    ax2 = ax.twinx()
    ax2.hist(centroids[:, 0], bins=20, density=True, alpha=0.5)
    ax2.set(ylabel="Density")
    plt.show()
