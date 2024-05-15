import itk.itkMinimumImageFilterPython
from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import label, binary_dilation
from skimage.measure import regionprops
import skimage.filters as filters
from skimage import graph   
from scipy import ndimage as ndi


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


def watershed_segmentation(image, markers, threshold):
    ws_regions = watershed(-image, markers)
    # Merge basins with boundary height less than the threshold
    ws_labels = np.unique(ws_regions)
    merged = np.zeros_like(ws_regions)
    edges = np.array(find_boundaries(ws_regions))*255
    edges = edges.astype(np.uint8)
    image = (image*255).astype(np.uint8)    
    rag = graph.rag_boundary(image, edges)
    rgb_image = np.dstack([image]*3)
    graph.show_rag(ws_regions, rag, rgb_image)
    plt.show()

    regions = regionprops(ws_regions)

    # Relabel the remaining basins
    ws_regions = label(ws_regions)
    
    return ws_regions

def process(image, high_contrast=None, sigma_blur=1, pixel_intensity_thresh=0.003, watershed_min_saliency=0.15, dot_intensity_thresh=0.05, size=50):
    # Get the mean value of the background
    mean_background = get_mean_region(image, high_contrast, "Background", size=size, vmax=None)
    # Get the mean value of the signal + background
    mean_signal_background = get_mean_region(image, high_contrast, "Signal + Background", size=size, vmax=None)
    # smooth using isotrpoic gaussian blur
    image = ndi.gaussian_filter(image, sigma=sigma_blur)
    normalized_image = (image - mean_background) / (image.max() - mean_background)
    # threshold the image
    thresholded_image = np.where(normalized_image > pixel_intensity_thresh, normalized_image, 0)
    # renormalize the image
    renormalized_image = (thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min())
    # identify image maxima 
    maxima = peak_local_max(renormalized_image, threshold_abs=dot_intensity_thresh)
    markers = np.zeros_like(renormalized_image)
    markers[tuple(maxima.T)] = np.arange(len(maxima)) + 1
    ws_labels = watershed_segmentation(renormalized_image, markers, watershed_min_saliency)
    return maxima
   