from cellpose.models import Cellpose
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
from skimage.measure import regionprops
from scipy import ndimage as ndi
import cv2 as cv
import bigfish.detection as detection
import pandas as pd

DEFAULT_HCR_PARAMS = {
    "sigma_blur": 0.2,
    "pixel_intensity_thresh": 0.0,
    "fg_width": 0.2,
    "dot_intensity_thresh": 0.05,
    "verbose": False,
}


# random colormap for labelling the regions
def random_cmap(n=256, name="random_cmap"):
    """Create a random colormap for labelling the regions."""
    np.random.seed(0)
    colors = np.random.rand(n, 4)
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)
    return cmap


def quantify_hcr(
    image,
    mean_background,
    sigma_blur=1,
    pixel_intensity_thresh=0.000,
    fg_width=0.2,
    dot_intensity_thresh=0.05,
    voxel_size=(100, 100),
    spot_radius=(150, 150),
    verbose=False,
):
    blurred = ndi.gaussian_filter(image, sigma=sigma_blur)
    normalized_image = (blurred - mean_background) / (blurred.max() - mean_background)
    thresholded_image = np.where(
        normalized_image > pixel_intensity_thresh, normalized_image, 0
    )
    renormalized_image = (thresholded_image - thresholded_image.min()) / (
        thresholded_image.max() - thresholded_image.min()
    )
    thresh = (renormalized_image > dot_intensity_thresh).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv.dilate(thresh, kernel, iterations=10)
    dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    if verbose:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].imshow(image, cmap="afmhot", vmax=np.mean(image) + 3 * np.std(image))
        ax[1].imshow(dist_transform, cmap="afmhot", vmax=dist_transform.max())
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
    if verbose:
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].imshow(image, cmap="afmhot", vmax=20 * np.mean(image))
        ax[1].imshow(labels, cmap=r_cmap)
        plt.show()
    props = regionprops(labels, intensity_image=renormalized_image)
    centroids = np.array([prop.centroid for prop in props])
    centroids_post_decomposition, dense_regions, reference_spot = (
        detection.decompose_dense(
            image=image.astype(np.uint16),
            spots=centroids,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
            alpha=0.90,  # alpha impacts the number of spots per candidate region
            beta=1,  # beta impacts the number of candidate regions to decompose
            gamma=5,  # gamma the filtering step to denoise the image
        )
    )
    centroids_post_decomposition = pd.DataFrame(
        centroids_post_decomposition, columns=["x", "y"]
    )
    return labels, centroids_post_decomposition


def quantify_hcr_bf(image, voxel_size=(103, 103), spot_radius=(150, 150), threshold=None):
    """Quantify the staining in the cells."""
    spots, threshold = detection.detect_spots(
        images=image,
        threshold=threshold,
        return_threshold=True,
        voxel_size=voxel_size,  # in nanometer (one value per dimension zyx)
        spot_radius=spot_radius,  # in nanometer (one value per dimension zyx)
    )
    centroids_post_decomposition, dense_regions, reference_spot = (
        detection.decompose_dense(
            image=image.astype(np.uint16),
            spots=spots,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
            alpha=0.90,  # alpha impacts the number of spots per candidate region
            beta=1,  # beta impacts the number of candidate regions to decompose
            gamma=5,  # gamma the filtering step to denoise the image
        )
    )
    centroids_post_decomposition = pd.DataFrame(
        centroids_post_decomposition, columns=["x", "y"]
    )
    return centroids_post_decomposition


def quantify_staining(image, masks, cell_data, name="staining"):
    """Quantify the staining in the cells."""
    intensities = []
    for label in cell_data.index:
        mask = masks == label
        intensities.append(image[mask].mean())
    cell_data[f"{name}_mean_intensity"] = intensities
    return cell_data
