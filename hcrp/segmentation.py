from cellpose.models import Cellpose
import numpy as np
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from hcrp import quantify_hcr
from hcrp.hcr import cellpose_hcr
import pandas as pd
from collections import OrderedDict
import cv2


def get_random_cmap(n_colors):
    random_colour_map = np.random.rand(n_colors, 3)
    cmap = LinearSegmentedColormap.from_list(name="random", colors=random_colour_map)
    return cmap


def get_cells(image, diameter=30, channels=[0, 0]):
    image = np.array(image)
    image = (image / 255).astype(np.uint8)
    model = Cellpose(gpu=True, model_type="cyto")
    masks, flows, styles, diams = model.eval(image, diameter=30, channels=[0, 0])
    props = regionprops(masks)
    return masks, props


def get_inernal_indices(centroids, roi):
    polygon = Polygon(roi)
    internal_indices = []
    for i, centroid in enumerate(centroids):
        if polygon.contains(Point(centroid)):
            internal_indices.append(i)
    return internal_indices


def get_spline_points(midline, contour, n_points=1000):
    tck, _ = splprep(midline.T, s=0)
    n_points = 1000
    t_values = np.linspace(0, 1, n_points)
    spline_points = np.array(splev(t_values, tck))
    spline_dist = np.zeros(n_points)
    for i in range(1, n_points):
        increment = np.linalg.norm(spline_points[:, i] - spline_points[:, i - 1])
        spline_dist[i] = spline_dist[i - 1] + increment
    return spline_points.T, spline_dist


default_hcr_params = {
    "sigma_blur": 0.2,
    "pixel_intensity_thresh": 0.0,
    "fg_width": 0.2,
    "dot_intensity_thresh": 0.05,
    "verbose": False,
}


def segment(
    stack_path,
    label_location,
    data_out=None,
    channel_names=["brk", "dpp", "pmad", "nuclear"],
    channel_types=["hcr", "hcr", "staining", "nuclear"],
    hcr_params=None,
    diameter=30,
):
    """This function takes in the path to a stack of images and the location of the midline, contour and background labels for the stack
    and returns a pandas dataframe with the location of the segmented cells and the mean intensity or count of the number of mrnas in each cell.
    It"""
    assert channel_names[-1] == "nuclear", "Last channel must be nuclear"
    if hcr_params is None:
        hcr_params = [default_hcr_params for _ in range(len(channel_names))]
    else:
        assert len(hcr_params) == len(
            channel_names
        ), "HCR Params must match non-nuclear channels"
    stack = imread(f"{stack_path}.tif")
    name = stack_path.split("/")[-1].split(".")[0]
    columns = ["spline_dist", "x", "y", "z"] + [
        f"{name}_count" if type == "hcr" else f"{name}_mean_intensity"
        for name, type in zip(channel_names[:-1], channel_types[:-1])
    ]
    data = pd.DataFrame(columns=columns)
    midline_data = pd.read_csv(f"{label_location}/{name}_midline.csv")
    z = midline_data["z"].iloc[0]
    midline = np.array(midline_data[["x", "y"]])
    contour = np.array(pd.read_csv(f"{label_location}/{name}_contour.csv")[["x", "y"]])
    background = pd.read_csv(f"{label_location}/{name}_background.csv", index_col=0)[
        "mean_intensity"
    ]
    signal_background = pd.read_csv(
        f"{label_location}/{name}_signal_background.csv", index_col=0
    )["mean_intensity"]
    spline_points, spline_dist = get_spline_points(midline, contour)
    image = stack[z, :, :, -1]
    masks, props = get_cells(image, diameter=diameter)
    centroids = np.array([prop.centroid for prop in props])
    cell_spline_indices = np.argmin(cdist(centroids, spline_points), axis=1)
    distal_index = np.argmin(spline_dist[cell_spline_indices])
    spline_dist = spline_dist - spline_dist[distal_index]
    internal_cell_indices = get_inernal_indices(centroids, contour)
    cell_closest_spline_indices = np.argmin(
        cdist(centroids[internal_cell_indices], spline_points), axis=1
    )
    cell_spline_dist = spline_dist[cell_closest_spline_indices]
    data["spline_dist"] = cell_spline_dist
    data["x"] = centroids[internal_cell_indices, 0]
    data["y"] = centroids[internal_cell_indices, 1]
    data["z"] = z
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
        channel = stack[z, :, :, i]
        channel_8bit = (channel / channel.max() * 255).astype(np.uint8)
        normalized_channel = cv2.equalizeHist(channel_8bit)
        if ctype == "hcr":
            mask, hcr_props = quantify_hcr(
                channel, background.loc[cname], **hcr_params[i]
            )
            # mask, hcr_props = cellpose_hcr(channel_8bit, diameter=5)
            hcr_centroids = np.array([prop.centroid for prop in hcr_props])
            internal_hit_indices = get_inernal_indices(hcr_centroids, contour)
            if hcr_params[i]["verbose"]:
                plt.imshow(
                    channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel)
                )
                # plt.imshow(normalized_channel, cmap="afmhot")
                plt.scatter(centroids[:, 1], centroids[:, 0], c="r")
                plt.scatter(
                    centroids[internal_cell_indices, 1],
                    centroids[internal_cell_indices, 0],
                    c="g",
                )
                plt.scatter(
                    hcr_centroids[internal_hit_indices, 1],
                    hcr_centroids[internal_hit_indices, 0],
                    c="y",
                )
                plt.plot(spline_points[:, 1], spline_points[:, 0], c="b")
                plt.show()
            distances_to_cells = cdist(
                hcr_centroids[internal_hit_indices], centroids[internal_cell_indices]
            )
            hcr_closest_cell_indices = np.argmin(distances_to_cells, axis=1)
            hcr_counts = np.bincount(hcr_closest_cell_indices, minlength=len(internal_cell_indices))
            data[f"{name}_count"] = hcr_counts
        else:
            internal_cell_masks = [masks == i for i in internal_cell_indices]
            staining_intensities = [
                np.mean(channel[mask]) for mask in internal_cell_masks
            ]
            data[f"{name}_mean_intensity"] = staining_intensities
    print(data)
    return masks, data
