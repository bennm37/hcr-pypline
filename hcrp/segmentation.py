from cellpose.models import Cellpose
import numpy as np
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
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


def get_internal_indices(centroids, roi):
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
    verbose=False,
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
        f"{cname}_count" if ctype == "hcr" else f"{cname}_mean_intensity"
        for cname, ctype in zip(channel_names[:-1], channel_types[:-1])
    ]
    data = pd.DataFrame(columns=columns)
    midline, contour, background, z = load_labels(label_location, name)
    image = stack[z, :, :, -1]
    spline_points, spline_dist = get_spline_points(midline, contour)
    masks, props = get_cells(image, diameter=diameter)
    n_cells = len(props)
    centroids = np.array([prop.centroid for prop in props])
    labels = [prop.label for prop in props]
    cell_spline_indices = np.argmin(cdist(centroids, spline_points), axis=1)
    distal_index = np.argmin(spline_dist[cell_spline_indices])
    spline_dist = spline_dist - spline_dist[distal_index]
    internal_cell_indices = get_internal_indices(centroids, contour)
    internal_masks = np.where(
        np.in1d(masks, internal_cell_indices).reshape(masks.shape), masks, 0
    )
    cell_closest_spline_indices = np.argmin(
        cdist(centroids[internal_cell_indices], spline_points), axis=1
    )
    cell_spline_dist = spline_dist[cell_closest_spline_indices]
    r_cmap = get_random_cmap(n_cells)
    data["spline_dist"] = cell_spline_dist
    data["x"] = centroids[internal_cell_indices, 0]
    data["y"] = centroids[internal_cell_indices, 1]
    data["z"] = z
    if verbose:
        plt.imshow(image, cmap="afmhot")
        plt.imshow(masks, alpha=0.8, cmap=r_cmap, vmin=0, vmax=n_cells)
        norm = Normalize(vmin=0, vmax=n_cells)
        plt.scatter(centroids[:, 1], centroids[:, 0], c=r_cmap(norm(labels)))
        plt.show()
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
        channel = stack[z, :, :, i]
        channel_8bit = (channel / channel.max() * 255).astype(np.uint8)
        normalized_channel = cv2.equalizeHist(channel_8bit)
        if ctype == "hcr":
            mask, hcr_centroids = quantify_hcr(
                channel, background.loc[cname], **hcr_params[i]
            )
            internal_hit_indices = get_internal_indices(hcr_centroids, contour)
            distances_to_cells = cdist(
                hcr_centroids[internal_hit_indices], centroids[internal_cell_indices]
            )
            hcr_closest_cell_indices = np.argmin(distances_to_cells, axis=1)
            hcr_counts = np.bincount(
                hcr_closest_cell_indices, minlength=len(internal_cell_indices)
            )
            if verbose:
                # show spline and centroids
                plt.imshow(
                    channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel)
                )
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
                # show hcr centroids and counts
                plt.imshow(channel, cmap="afmhot")
                plt.scatter(
                    centroids[internal_cell_indices, 1],
                    centroids[internal_cell_indices, 0],
                    c=hcr_counts,
                )
                plt.imshow(
                    internal_masks,
                    cmap=get_random_cmap(len(internal_cell_indices)),
                    alpha=0.3,
                )
                plt.show()
            data[f"{cname}_count"] = hcr_counts
        else:
            staining_intensities = np.array(
                [np.mean(channel[masks == i]) for i in labels]
            )
            internal_staining_intensities = staining_intensities[internal_cell_indices]
            if verbose:
                plt.imshow(channel, cmap="afmhot")
                plt.scatter(
                    centroids[internal_cell_indices, 1],
                    centroids[internal_cell_indices, 0],
                    c=staining_intensities,
                )
                plt.imshow(
                    internal_masks,
                    cmap=get_random_cmap(len(internal_cell_indices)),
                    alpha=0.3,
                )
                plt.show()
            data[f"{cname}_mean_intensity"] = internal_staining_intensities
    return masks, data


def load_labels(label_location, name, midline_data):
    midline_data = pd.read_csv(f"{label_location}/{name}_midline.csv")
    z = midline_data["z"].iloc[0]
    midline = np.array(midline_data[["x", "y"]])
    contour = np.array(pd.read_csv(f"{label_location}/{name}_contour.csv")[["x", "y"]])
    background = pd.read_csv(f"{label_location}/{name}_background.csv", index_col=0)[
        "mean_intensity"
    ]
    return midline, contour, background, z


def aggregate(x, y, bin_size):
    bins = np.arange(np.min(x), np.max(x), bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(bins)
    y_binned = np.zeros(n_bins - 1)
    y_err_binned = np.zeros(n_bins - 1)
    for i in range(1, n_bins):
        indices = np.where(np.logical_and(x > bins[i - 1], x < bins[i]))[0]
        y_in_bin = [y[j] for j in indices]
        y_binned[i - 1] = np.mean(y_in_bin)
        y_err_binned[i - 1] = np.std(y_in_bin)
    return bin_centers, y_binned, y_err_binned
