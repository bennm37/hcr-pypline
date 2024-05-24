from cellpose.models import Cellpose
import numpy as np
from skimage import measure
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from hcrp import quantify_hcr, quantify_hcr_bf, quantify_staining, DEFAULT_HCR_PARAMS
from hcrp.labelling import load_labels_safe
import pandas as pd
from collections import OrderedDict
import cv2
import os


def process_layer(
    z,
    folder,
    filename,
    label_location,
    results_path=None,
    channel_names=["brk", "dpp", "pmad", "nuclear"],
    channel_types=["hcr", "hcr", "staining", "nuclear"],
    hcr_params=None,
    diameter=30,
    bf=True,
):
    """This function takes in the path to a stack of images and the location of the midline, contour and background labels for the stack
    and returns a pandas dataframe with the location of the segmented cells and the mean intensity or count of the number of mrnas in each cell.
    It"""
    assert channel_names[-1] == "nuclear", "Last channel must be nuclear"
    if hcr_params is None:
        hcr_params = [DEFAULT_HCR_PARAMS for _ in range(len(channel_names))]
    else:
        assert len(hcr_params) == len(
            channel_names
        ), "HCR Params must match non-nuclear channels"
    stack = imread(f"{folder}/{filename}")
    name = filename.split(".")[0]
    midline, contour, background, _, endoderm = load_labels_safe(folder, label_location, filename, endoderm=True)
    layer = stack[z]
    masks, cell_data = get_cell_data(layer[:, :, 3], diameter=diameter, polygon=contour)
    cell_data["z"] = z
    hcr_data = [None for _ in channel_names]
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
        if ctype == "hcr":
            if bf:
                hcr_data[i] = quantify_hcr_bf(layer[:, :, i], threshold=hcr_params[i]["dot_intensity_thresh"])
            else:
                hcr_masks, hcr_data[i] = quantify_hcr(
                    layer[:, :, 0], background[cname], **hcr_params[i]
                )
            hcr_data[i] = remove_external(hcr_data[i], contour)
            hcr_data[i], cell_data = project_to_cells(
                hcr_data[i], cell_data, name=cname
            )
            hcr_data[i] = project_to_midline(
                hcr_data[i], midline, contour, mesoderm_cutoff=endoderm
            )
        else:
            cell_data = quantify_staining(layer[:, :, 2], masks, cell_data, name=cname)
    cell_data = project_to_midline(
        cell_data, midline, contour, mesoderm_cutoff=endoderm
    )
    if results_path is not None:
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        cell_data.to_csv(f"{results_path}/{name}_cell_data_{z}.csv")
        np.save(f"{results_path}/{name}_masks_{z}.npy", masks)
        for i, cname in enumerate(channel_names[:-1]):
            if hcr_data[i] is not None:
                hcr_data[i].to_csv(f"{results_path}/{name}_{cname}_hcr_data_{z}.csv")
        print(f"Saved results to {results_path}")
    return masks, cell_data, hcr_data


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
    cell_data, midline, contour, n_points=1000, mesoderm_cutoff=None
):
    spline_points, spline_dist = get_spline_points(midline, contour)
    centroids = cell_data[["x", "y"]].values
    cell_spline_indices = np.argmin(cdist(centroids, spline_points), axis=1)
    cell_splines_offsets = np.min(cdist(centroids, spline_points), axis=1)
    distal_index = np.argmin(spline_dist[cell_spline_indices])
    spline_dist = spline_dist - spline_dist[cell_spline_indices[distal_index]]
    cell_spline_dist = spline_dist[cell_spline_indices]
    cell_data["spline_dist"] = cell_spline_dist
    if isinstance(mesoderm_cutoff, tuple):
        cell_data["endoderm"] = np.logical_and(
            cell_splines_offsets < mesoderm_cutoff[0], cell_spline_dist > mesoderm_cutoff[1]
        )
        cell_data["ectoderm"] = np.logical_not(cell_data["endoderm"])
    elif isinstance(mesoderm_cutoff, np.ndarray):
        inds = get_internal_indices(centroids, mesoderm_cutoff)
        cell_data["endoderm"] = np.in1d(cell_data.index, cell_data.index[inds])
        cell_data["ectoderm"] = np.logical_not(cell_data["endoderm"])
    return cell_data


def project_to_cells(hcr_data, cell_data, name):
    hcr_hits = hcr_data[["x", "y"]].values
    cell_centroids = cell_data[["x", "y"]].values
    distances = cdist(hcr_hits, cell_centroids)
    closest_cells = cell_data.index[np.argmin(distances, axis=1)]
    hcr_data[f"closet_cell_label"] = closest_cells
    cell_data[f"{name}_count"] = np.bincount(
        closest_cells, minlength=len(cell_data) + 1
    )[1:]
    return hcr_data, cell_data


def aggregate(x, y, bin_size, xmin=None, nan_value=np.nan, err_type="std"):
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
    if err_type == "std_err":
        y_err_binned = y_err_binned / np.sqrt(len(y_in_bin))
    else:
        y_err_binned = y_err_binned
    return bin_centers, y_binned, y_err_binned
