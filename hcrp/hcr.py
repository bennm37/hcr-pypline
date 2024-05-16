from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
import pandas as pd
from skimage.graph._rag import _edge_generator_from_csr, RAG
from skimage import graph
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import label, dilation, reconstruction
from skimage.measure import regionprops
import skimage.filters as filters
from scipy import ndimage as ndi
from scipy import sparse
import cv2 as cv


# random colormap for labelling the regions
def random_cmap(n=256, name="random_cmap"):
    """Create a random colormap for labelling the regions."""
    np.random.seed(0)
    colors = np.random.rand(n, 4)
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)
    return cmap


class FixedSizeRectangleSelector(RectangleSelector):
    def __init__(self, ax, onselect, size=50, **kwargs):
        super().__init__(ax, onselect, **kwargs)
        self.extents = 0, size, 0, size
        self._allow_creation = False
        self.update()

    def _onmove(self, event):
        """
        Motion notify event handler.

        This class only allows:
        - Translate
        """
        eventpress = self._eventpress
        # The calculations are done for rotation at zero: we apply inverse
        # transformation to events except when we rotate and move
        move = self._active_handle == "C"

        xdata, ydata = self._get_data_coords(event)
        dx = xdata - eventpress.xdata
        dy = ydata - eventpress.ydata
        x0, x1, y0, y1 = self._extents_on_press
        if move:
            x0, x1, y0, y1 = self._extents_on_press
            dx = xdata - eventpress.xdata
            dy = ydata - eventpress.ydata
            x0 += dx
            x1 += dx
            y0 += dy
            y1 += dy
        self.extents = x0, x1, y0, y1


def get_mean_region(image, high_contrast, name, size=50, vmax=None):
    """Allow the user to to select the location of a rectangular region of interest in an image and return the mean value of the region.
    The selector should be of a fixed size and the user should be able to move it around the image.
    """
    fig, ax = plt.subplots()
    if high_contrast is None:
        high_contrast = image
    ax.imshow(high_contrast, cmap="afmhot")
    ax.set_title(f"Select a region of interest for {name}")
    means = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        region = image[y1:y2, x1:x2]
        means.append(np.mean(region))
        print(f"Mean value of {name}: {means[-1]}")

    selector = FixedSizeRectangleSelector(
        ax,
        onselect,
        size=size,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )

    def reset(event):
        if event.key == "r":
            means.clear()
            selector.extents = 0, size, 0, size
            selector.update()
            ax.set_title(f"Select a region of interest for {name}")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", reset)
    plt.show()
    return means[-1]

def quantify_hcr(
    image,
    high_contrast=None,
    sigma_blur=1,
    pixel_intensity_thresh=0.003,
    fg_width=0.2,
    dot_intensity_thresh=0.05,
    size=50,
    verbose=False,
):
    # Get the mean value of the background
    mean_background = get_mean_region(image, high_contrast, "Background", size=size, vmax=None)
    # # Get the mean value of the signal + background
    mean_signal_background = get_mean_region(image, high_contrast, "Signal + Background", size=size, vmax=None)
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
    erode = cv.erode(thresh, np.ones((2,2)), iterations=1)
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