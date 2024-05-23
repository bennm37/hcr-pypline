
from cellpose.models import Cellpose
from hcrp import *
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
dropbox = "/Users/huanga/The Francis Crick Dropbox/VincentJ"
folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset3_tkv_dad_dpp_Nuclei/"
# folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset3_tkv_dad_dpp_Nuclei/"
# filename = "Td_Hoechst_dad488_dpp546_tkv647_20240521_Embryo01_T1.tif"
filename = "Td_Hoechst_dad488_dpp546_tkv647_20240521_Embryo01_T2.tif"
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
# if not os.path.exists(label_location):
#     os.makedirs(label_location)
#     label(f"{folder}/{filename}", label_location, channel_names=["tkv", "dpp", "dad", "nuclear"])
# label(f"{folder}/{filename}", label_location, channel_names=["tkv", "dpp", "dad", "nuclear"])
stack = imread(f"{folder}/{filename}")

midline, contour, background, z_midline = load_labels(label_location, filename)
def get_data(z):
    masks, cell_data = get_cell_data(stack[z, :, :, 3], diameter=30, polygon=contour)
    tkv_masks, tkv_data = quantify_hcr(stack[z, :, :, 0], background["tkv"], **tkv_params)
    tkv_data = remove_external(tkv_data, contour)
    dpp_masks, dpp_data = quantify_hcr(stack[z, :, :, 1], background["dpp"], **dpp_params)
    dpp_data = remove_external(dpp_data, contour)
    dad_masks, dad_data = quantify_hcr(stack[z, :, :, 2], background["dad"], **dad_params)
    dad_data = remove_external(dad_data, contour)
    tkv_data, cell_data = project_to_cells(tkv_data, cell_data, name="tkv")
    dpp_data, cell_data = project_to_cells(dpp_data, cell_data, name="dpp")
    dad_data, cell_data = project_to_cells(dad_data, cell_data, name="dad")
    cell_data = project_to_midline(cell_data, midline, contour, mesoderm_cutoff=(50, 50))
    return cell_data, tkv_data, dpp_data, dad_data

def plot_error_bar(ax, cell_data, bin_size, cname, color, type):
    bin_centers, c_mean, c_error = aggregate(
                cell_data["spline_dist"], cell_data[f"{cname}_count"], bin_size, xmin=0
            )
    label = f"{cname}_{type}"
    if type=="ectoderm":
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



z_range = np.array(range(z_midline-2, z_midline+3))
results = {z:None for z in z_range}
# z_range = np.array([z_midline, z_midline+1])
fig, ax = plt.subplots(3,1) 
cmap = plt.get_cmap("viridis")   
norm = plt.Normalize(vmin=z_range.min(), vmax=z_range.max())
bin_size = 50
for z in z_range:
    print(z)
    cell_data, tkv_data, dpp_data, dad_data = get_data(z)
    results[z] = cell_data
    for i, cname in enumerate(["tkv","dpp","dad"]):
        ax[i].set(xlabel="Spline Distance (px)")
        ax[i].set(ylabel=f"{cname}")
        plot_error_bar(ax[i], cell_data[cell_data["endoderm"]], bin_size, cname, cmap(norm(z)), "endoderm")
        plot_error_bar(ax[i], cell_data[cell_data["ectoderm"]], bin_size, cname, cmap(norm(z)), "ectoderm")
plt.savefig(f"data/{filename}_multiple_z.pdf")
plt.show()

fig, ax = plt.subplots(3,1)
cell_data = pd.concat([result for result in results.values()])
for i, cname in enumerate(["tkv","dpp","dad"]):
    ax[i].set(xlabel="Spline Distance (px)")
    ax[i].set(ylabel=f"{cname}")
    plot_error_bar(ax[i], cell_data[cell_data["endoderm"]], bin_size, cname, "blue", "endoderm")
    plot_error_bar(ax[i], cell_data[cell_data["ectoderm"]], bin_size, cname, "blue", "ectoderm")
plt.savefig(f"data/{filename}_average_z.pdf")
plt.show()
