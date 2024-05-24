from cellpose.models import Cellpose
from hcrp import *
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# setup
folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset3_tkv_dad_dpp_Nuclei"
filename = "Td_Hoechst_dad488_dpp546_tkv647_20240521_Embryo01_T1.tif"
label_location = "data/tkv_dpp_dad_test"
tkv_params = DEFAULT_HCR_PARAMS.copy()
tkv_params["dot_intensity_thresh"] = 0.075
tkv_params["sigma_blur"] = 1
tkv_params["verbose"] = False
dpp_params = DEFAULT_HCR_PARAMS.copy()
dpp_params["dot_intensity_thresh"] = 0.075
dpp_params["sigma_blur"] = 0.2
dpp_params["verbose"] = False
dad_params = DEFAULT_HCR_PARAMS.copy()
dad_params["dot_intensity_thresh"] = 0.075
dad_params["sigma_blur"] = 0.2
dad_params["verbose"] = False
stack = imread(f"{folder}/{filename}")

midline, contour, background, z_midline = load_labels_safe(
    folder, label_location, filename
)


def plot_error_bar(ax, cell_data, bin_size, cname, color, type):
    bin_centers, c_mean, c_error = aggregate(
        cell_data["spline_dist"], cell_data[f"{cname}_count"], bin_size, xmin=0
    )
    label = f"{cname}_{type}"
    if type == "ectoderm":
        linestyle = "--"
    else:
        linestyle = "-"
    err = ax.errorbar(
        bin_centers,
        c_mean,
        yerr=c_error,
        label=label,
        linestyle=linestyle,
        color=color,
        capsize=5,
    )



z_range = np.array(range(z_midline - 2, z_midline + 3))
results = {z: None for z in z_range}
# z_range = np.array([z_midline, z_midline+1])
fig, ax = plt.subplots(3, 1)
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=z_range.min(), vmax=z_range.max())
bin_size = 50
for z in z_range:
    print(z)
    cell_data, hcr_data = process_layer(
        stack[z],
        midline,
        contour,
        background,
        results_path="data/tkv_dpp_dad_test",
        channel_names=["tkv", "dpp", "dad", "nuclear"],
        channel_types=["hcr", "hcr", "hcr", "nuclear"],
        hcr_params=[tkv_params, dpp_params, dad_params],
        diameter=30,
        verbose=False,
    )
    results[z] = cell_data
    for i, cname in enumerate(["tkv", "dpp", "dad"]):
        ax[i].set(xlabel="Spline Distance (px)")
        ax[i].set(ylabel=f"{cname}")
        plot_error_bar(
            ax[i],
            cell_data[cell_data["endoderm"]],
            bin_size,
            cname,
            cmap(norm(z)),
            "endoderm",
        )
        plot_error_bar(
            ax[i],
            cell_data[cell_data["ectoderm"]],
            bin_size,
            cname,
            cmap(norm(z)),
            "ectoderm",
        )
plt.savefig(f"data/{filename}_multiple_z.pdf")
plt.show()

fig, ax = plt.subplots(3, 1)
# cell_data = pd.concat([result for result in results.values()])
for i, cname in enumerate(["tkv", "dpp", "dad"]):
    ax[i].set(xlabel="Spline Distance (px)")
    ax[i].set(ylabel=f"{cname}")
    plot_error_bar(
        ax[i], cell_data[cell_data["endoderm"]], bin_size, cname, "blue", "endoderm"
    )
    plot_error_bar(
        ax[i], cell_data[cell_data["ectoderm"]], bin_size, cname, "blue", "ectoderm"
    )
plt.savefig(f"data/{filename}_average_z.pdf")
plt.show()
