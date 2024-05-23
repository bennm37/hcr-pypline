from cellpose.models import Cellpose
import numpy as np
from skimage import measure
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from hcrp import quantify_hcr, quantify_staining
from hcrp.labelling import load_labels
import pandas as pd
from collections import OrderedDict
import cv2


def get_random_cmap(n_colors):
    random_colour_map = np.random.rand(n_colors, 3)
    cmap = LinearSegmentedColormap.from_list(name="random", colors=random_colour_map)
    return cmap


def get_cell_data(image, diameter=30, channels=[0, 0], polygon=None):
    image = np.array(image)
    image = (image / 255).astype(np.uint8)
    model = Cellpose(gpu=True, model_type="cyto")
    masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=[0, 0])
    if polygon is not None:
        cell_data = pd.DataFrame(
            measure.regionprops_table(masks, properties=("label", "centroid"))
        )
        centroids = cell_data[["centroid-0", "centroid-1"]].values
        labels = cell_data["label"].values
        internal_indices = get_internal_indices(centroids, polygon)
        internal_masks = np.where(
            np.in1d(masks, labels[internal_indices]).reshape(masks.shape), masks, 0
        )
        masks = measure.label(internal_masks)  # relabel the masks
    cell_data = pd.DataFrame(
        measure.regionprops_table(masks, properties=("label", "centroid", "area"))
    )
    cell_data.rename(columns={"centroid-0": "x", "centroid-1": "y"}, inplace=True)
    cell_data.set_index("label", inplace=True)
    return masks, cell_data


def get_internal_indices(centroids, polygon):
    polygon = Polygon(polygon)
    internal_indices = []
    for i, centroid in enumerate(centroids):
        if polygon.contains(Point(centroid)):
            internal_indices.append(i)
    return internal_indices


def remove_external(dataframe, polygon):
    centroids = dataframe[["x", "y"]].values
    internal_indices = get_internal_indices(centroids, polygon)
    return dataframe.iloc[internal_indices]


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


def project_to_midline(
    cell_data, midline, contour, n_points=1000, mesoderm_cutoff=(50, 50)
):
    spline_points, spline_dist = get_spline_points(midline, contour)
    centroids = cell_data[["x", "y"]].values
    cell_spline_indices = np.argmin(cdist(centroids, spline_points), axis=1)
    cell_splines_offsets = np.min(cdist(centroids, spline_points), axis=1)
    distal_index = np.argmin(spline_dist[cell_spline_indices])
    spline_dist = spline_dist - spline_dist[cell_spline_indices[distal_index]]
    cell_spline_dist = spline_dist[cell_spline_indices]
    cell_data["spline_dist"] = cell_spline_dist
    cell_data["endoderm"] = np.logical_and(
        cell_splines_offsets < mesoderm_cutoff[0], cell_spline_dist > mesoderm_cutoff[1]
    )
    cell_data["ectoderm"] = np.logical_not(cell_data["endoderm"])
    return cell_data


def project_to_cells(hcr_data, cell_data, name):
    hcr_hits = hcr_data[["x", "y"]].values
    cell_centroids = cell_data[["x", "y"]].values
    distances = cdist(hcr_hits, cell_centroids)
    closest_cells = cell_data.index[np.argmin(distances, axis=1)]
    hcr_data[f"closet_cell_label"] = closest_cells
    cell_data[f"{name}_count"] = np.bincount(closest_cells, minlength=len(cell_data)+1)[
        1:
    ]
    return hcr_data, cell_data


def aggregate(x, y, bin_size, xmin=None, nan_value=np.nan):
    if xmin is None:
        xmin = np.min(x)
    bins = np.arange(xmin, np.max(x), bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(bins)
    y_binned = np.zeros(n_bins - 1)
    y_err_binned = np.zeros(n_bins - 1)
    for i in range(1, n_bins):
        indices = np.where(np.logical_and(x > bins[i - 1], x < bins[i]))[0]
        y_in_bin = [y.values[j] for j in indices]
        if len(y_in_bin) == 0:
            y_binned[i - 1] = nan_value
            y_err_binned[i - 1] = nan_value
        else:
            y_binned[i - 1] = np.mean(y_in_bin)
            y_err_binned[i - 1] = np.std(y_in_bin)
    return bin_centers, y_binned, y_err_binned


def segment(brk_params, dpp_params, stack, midline, contour, background, z):
    masks, cell_data = get_cell_data(stack[z, :, :, 3], diameter=30, polygon=contour)
    cell_data = quantify_staining(stack[z, :, :, 2], masks, cell_data, name="pmad")
    brk_masks, brk_data = quantify_hcr(
        stack[z, :, :, 0], background["brk"], **brk_params
    )
    brk_data = remove_external(brk_data, contour)
    dpp_masks, dpp_data = quantify_hcr(
        stack[z, :, :, 1], background["dpp"], **dpp_params
    )
    dpp_data = remove_external(dpp_data, contour)
    brk_data, cell_data = project_to_cells(brk_data, cell_data, name="brk")
    dpp_data, cell_data = project_to_cells(dpp_data, cell_data, name="dpp")
    cell_data = project_to_midline(cell_data, midline, contour, mesoderm_cutoff=(50, 0))
    return masks, cell_data, brk_data, dpp_data
