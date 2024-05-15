from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, maximum_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import label, binary_dilation, reconstruction
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

def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count,
    }

def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass

def watershed_segmentation(image, markers, threshold):
    ws_labels = watershed(-image, markers, compactness=0.0001)
    plt.imshow(ws_labels)
    plt.show()
    # Merge basins with boundary height less than the threshold
    edge_map = filters.sobel(image)
    # edge_map = find_boundaries(ws_labels, mode="inner", connectivity=1)
    ws_labels = ws_labels.astype(np.uint8)
    # image = (image*255).astype(np.uint8)    
    rag = graph.rag_boundary(ws_labels, edge_map.astype(float))
    rgb_image = np.dstack([image]*3)
    normalize_image = Normalize(vmin=image.min(), vmax=np.mean(image)*30)
    rgb_image = plt.get_cmap("afmhot")(normalize_image(image))[:,:,0:3]
    graph.show_rag(ws_labels, rag, np.dstack([ws_labels]*3), img_cmap="YlOrBr", edge_cmap="coolwarm", edge_width=1.5)
    plt.show()
    merged = graph.merge_hierarchical(ws_labels, rag, threshold, rag_copy=True, in_place_merge=True, merge_func=merge_boundary, weight_func=weight_boundary)
    label(merged)

    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax[0].imshow(image, cmap="afmhot", vmax=20*np.mean(image))
    ax[1].imshow(merged)
    plt.show()
    return merged

def process(image, high_contrast=None, sigma_blur=1, pixel_intensity_thresh=0.003, watershed_min_saliency=0.15, dot_intensity_thresh=0.05, size=50):
    # Get the mean value of the background
    # mean_background = get_mean_region(image, high_contrast, "Background", size=size, vmax=None)
    # # Get the mean value of the signal + background
    # mean_signal_background = get_mean_region(image, high_contrast, "Signal + Background", size=size, vmax=None)
    mean_background = 5.8792
    mean_signal_background = 8.1856
    # smooth using isotrpoic gaussian blur
    image = ndi.gaussian_filter(image, sigma=sigma_blur)
    normalized_image = (image - mean_background) / (image.max() - mean_background)
    # threshold the image
    thresholded_image = np.where(normalized_image > pixel_intensity_thresh, normalized_image, 0)
    # renormalize the image
    renormalized_image = (thresholded_image - thresholded_image.min()) / (thresholded_image.max() - thresholded_image.min())
    # calculate hdome
    seed = np.copy(renormalized_image)
    seed[1:-1, 1:-1] = renormalized_image.min()
    mask = renormalized_image
    # Skimage regional max
    # dilated = reconstruction(seed, mask, method='dilation')
    # hdome = renormalized_image - dilated
    # maxima = peak_local_max(hdome)
    neighborhood_structure = generate_binary_structure(2, 2)
    regional_max = maximum_filter(renormalized_image, footprint=neighborhood_structure)==renormalized_image
    markers = label(regional_max)
    plt.imshow(renormalized_image, cmap="afmhot", vmax=20*np.mean(renormalized_image))
    plt.imshow(markers, alpha=0.5)
    plt.show()
    ws_labels = watershed_segmentation(renormalized_image, markers, watershed_min_saliency)
    props = regionprops(ws_labels, intensity_image=renormalized_image)
    # remove dim objects 
    ws_labels = np.where(props[0].mean_intensity > dot_intensity_thresh, ws_labels, 0)
    plt.hist(props[0].mean_intensity, bins=1000, density=True)
    plt.hist(renormalized_image.flatten(), bins=1000, alpha=0.5, density=True)
    plt.show()
    ws_labels = label(ws_labels)
    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax[0].imshow(image, cmap="afmhot", vmax=20*np.mean(image))
    ax[1].imshow(ws_labels)
    plt.show()
    return ws_labels
   