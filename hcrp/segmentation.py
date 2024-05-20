from cellpose.models import Cellpose
import numpy as np
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import euclidean
from hcrp import quantify_hcr
import pandas as pd
from collections import OrderedDict
import cv2


def get_random_cmap(n_colors):
    random_colour_map = np.random.rand(n_colors, 3)
    cmap = LinearSegmentedColormap.from_list(name="random", colors=random_colour_map)
    return cmap


def get_cells(image, diameter=30, channels=[0, 0]):
    image = np.array(image)
    image = (image/255).astype(np.uint8)
    model = Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(image, diameter=30, channels=[0,0])
    props = regionprops(masks)
    return masks, props


def get_inernal_indices(centroids, roi):
    polygon = Polygon(roi)
    internal_indices = []
    for i, centroid in enumerate(centroids):
        if polygon.contains(Point(centroid)):
            internal_indices.append(i)
    return internal_indices


def segment(
    stack_path,
    label_location,
    data_out=None,
    channel_names=["brk","dpp","pmad","nuclear"],
    channel_types=["hcr","hcr","staining","nuclear"],
    hcr_params={
        "sigma_blur": 1,
        "pixel_intensity_thresh": 0.0,
        "fg_width": 0.2,
        "dot_intensity_thresh": 0.05,
        "verbose": False,
    },
    diameter=30,
):
    assert channel_names[-1] == "nuclear", "Last channel must be nuclear"
    stack = imread(f"{stack_path}.tif")
    name = stack_path.split("/")[-1].split(".")[0]
    columns = ["midline_dist", "x", "y", "z"] + [
        f"{name}_mean_intensity" for name in channel_names[:-1]
    ]
    data = pd.DataFrame(columns=columns)
    midline_data = pd.read_csv(f"{label_location}/{name}_midline.csv")
    z = midline_data["z"].iloc[0]
    midline = np.array(midline_data[["x", "y"]])
    contour = np.array(pd.read_csv(f"{label_location}/{name}_contour.csv")[["x", "y"]])
    background = pd.read_csv(f"{label_location}/{name}_background.csv", index_col=0)["mean_intensity"]
    signal_background = pd.read_csv(f"{label_location}/{name}_signal_background.csv", index_col=0)[
        "mean_intensity"
    ]
    image = stack[z, :, :, -1]
    masks, props = get_cells(image, diameter=diameter)
    centroids = np.array([prop.centroid for prop in props])
    internal_cell_indices = get_inernal_indices(centroids, contour)
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
        channel = stack[z, :, :, i]
        # normalize channel

        channel_8bit = (channel / 2**16 * 255).astype(np.uint8)
        normalized_channel = cv2.equalizeHist(channel_8bit)
        if ctype == "hcr":
            mask, hcr_props = quantify_hcr(channel, background.loc[cname], **hcr_params)
            hcr_centroids = np.array([prop.centroid for prop in hcr_props])
            internal_hit_indices = get_inernal_indices(hcr_centroids, contour)
            plt.imshow(normalized_channel, cmap="afmhot", vmax=300)
            plt.scatter(centroids[:,1], centroids[:,0], c="r")
            plt.scatter(centroids[internal_cell_indices, 1], centroids[internal_cell_indices, 0], c="g")
            plt.scatter(hcr_centroids[:,1], hcr_centroids[:,0], c="b")
            plt.scatter(hcr_centroids[internal_hit_indices, 1], hcr_centroids[internal_hit_indices, 0], c="y")
            plt.plot(contour[:, 1], contour[:, 0], c="r")
            plt.show()
        else:
            internal_cell_masks = [masks==i for i in internal_cell_indices]
            pmad_intensities = [np.mean(channel[mask]) for mask in internal_cell_masks]
            
    return masks
