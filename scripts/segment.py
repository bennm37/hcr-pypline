from cellpose.models import Cellpose
from hcrp import *

# from hcrp.plotting import plot_gradients, plot_channels
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import numpy as np

# dropbox = "/Users/nicholb/Dropbox"
# folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
filename = "Stg01_Emb03_T101.tif"
label_location = "data/Limb_Ext_Stg01_test"
brk_params = DEFAULT_HCR_PARAMS.copy()
brk_params["dot_intensity_thresh"] = 0.075
brk_params["sigma_blur"] = 1
brk_params["verbose"] = False
dpp_params = DEFAULT_HCR_PARAMS.copy()
dpp_params["dot_intensity_thresh"] = 0.075
dpp_params["sigma_blur"] = 0.2
dpp_params["verbose"] = False
if not os.path.exists(label_location):
    os.makedirs(label_location)
    label(f"{folder}/{filename}", label_location)
stack = imread(f"{folder}/{filename}")

midline, contour, background, z_midline = load_labels(label_location, filename)
z = z_midline - 1
masks, cell_data = get_cell_data(stack[z, :, :, 3], diameter=30, polygon=contour)
cell_data = quantify_staining(stack[z, :, :, 2], masks, cell_data, name="pmad")
brk_masks, brk_data = quantify_hcr(stack[z, :, :, 0], background["brk"], **brk_params)
brk_data = remove_external(brk_data, contour)
dpp_masks, dpp_data = quantify_hcr(stack[z, :, :, 1], background["dpp"], **dpp_params)
dpp_data = remove_external(dpp_data, contour)
brk_data, cell_data = project_to_cells(brk_data, cell_data, name="brk")
dpp_data, cell_data = project_to_cells(dpp_data, cell_data, name="dpp")
cell_data = project_to_midline(cell_data, midline, contour, mesoderm_cutoff=(50, 0))

plot_hcr(brk_data, stack[z, :, :, 0])
plot_hcr(dpp_data, stack[z, :, :, 1])
plot_hcr_cell_projection(brk_data, cell_data, masks, masks)
plot_hcr_cell_projection(dpp_data, cell_data, masks, masks)
plot_cell_property(cell_data, stack[z, :, :, 2], "pmad_mean_intensity")
plot_cell_property(cell_data, stack[z, :, :, 2], "brk_count")
plot_cell_property(cell_data, stack[z, :, :, 2], "dpp_count")
plot_cell_property(cell_data, stack[z, :, :, 3], "endoderm")
plot_gradients(
    ["brk", "dpp", "pmad", "nuclear"],
    ["hcr", "hcr", "staining", "nuclear"],
    cell_data[cell_data["endoderm"]],
)
plt.show()
