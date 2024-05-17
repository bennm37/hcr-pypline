import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
from skimage.measure import regionprops
from scipy import ndimage as ndi
import cv2 as cv
from hcrp.labelling import get_mean_region


# random colormap for labelling the regions
def random_cmap(n=256, name="random_cmap"):
    """Create a random colormap for labelling the regions."""
    np.random.seed(0)
    colors = np.random.rand(n, 4)
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)
    return cmap


def quantify_hcr(
    image,
    high_contrast=None,
    sigma_blur=1,
    pixel_intensity_thresh=0.000,
    fg_width=0.2,
    dot_intensity_thresh=0.05,
    size=50,
    verbose=False,
):
    # Get the mean value of the background
    mean_background = get_mean_region(
        image, high_contrast, "Background", size=size, vmax=None
    )
    # # Get the mean value of the signal + background
    mean_signal_background = get_mean_region(
        image, high_contrast, "Signal + Background", size=size, vmax=None
    )
    # mean_background = 5.8792
    # mean_signal_background = 8.1856
    image = ndi.gaussian_filter(image, sigma=sigma_blur)
    normalized_image = (image - mean_background) / (image.max() - mean_background)
    # threshold the image
    thresholded_image = np.where(
        normalized_image > pixel_intensity_thresh, normalized_image, 0
    )
    # renormalize the image
    renormalized_image = (thresholded_image - thresholded_image.min()) / (
        thresholded_image.max() - thresholded_image.min()
    )
    thresh = (renormalized_image > dot_intensity_thresh).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(thresh, kernel, iterations=10)
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    if verbose:
        plt.imshow(dist_transform)
        plt.show()
    erode = cv.erode(thresh, np.ones((2, 2)), iterations=1)
    ret, sure_fg = cv.threshold(dist_transform, fg_width * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    if verbose:
        plt.imshow(sure_bg)
        plt.show()
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = markers.astype(np.int32)
    rgb_image = np.dstack([image * 255] * 3).astype(np.uint8)
    labels = cv.watershed(rgb_image, markers)
    r_cmap = random_cmap(labels.max())
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].imshow(image, cmap="afmhot", vmax=20 * np.mean(image))
    ax[1].imshow(labels, cmap=r_cmap)
    plt.show()
    props = regionprops(labels, intensity_image=renormalized_image)
    return labels, props
