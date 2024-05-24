from cellpose.models import Cellpose
from hcrp import *
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_all(folder, filename, label_location, results_path):
    stack = imread(f"{folder}/{filename}")
    name = filename.split(".")[0]
    midline, contour, background, z_midline = load_labels_safe(
        folder, label_location, filename
    )
    z = z_midline
    cell_data = pd.read_csv(f"{results_path}/{name}_cell_data_z.csv")
    brk_data = pd.read_csv(f"{results_path}/{name}_brk_hcr_data_z.csv")
    dpp_data = pd.read_csv(f"{results_path}/{name}_dpp_hcr_data_z.csv")
    masks = np.load(f"{results_path}/{name}_masks_z.npy")
    cell_data = pd.read_csv(f"{results_path}/{name}_cell_data_z.csv")
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
        pixel_to_mu=1 / 2.8906,
    )
    plt.show()

# CHANGE THIS BIT
folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
filename = "Stg01_Emb03_T101.tif"
label_location = "data/Limb_Ext_Stg01_test"
channel_names = ["brk", "dpp", "pmad", "nuclear"]
channel_types = ["hcr", "hcr", "staining", "nuclear"]
results_path = f"data/{filename.split('.')[0]}_results"
# EDIT HCR PARAMETERS IF NEEDED
params = DEFAULT_HCR_PARAMS.copy()
params["dot_intensity_thresh"] = 0.075
params["sigma_blur"] = 1
params["verbose"] = False

run = True # CHANGE THIS TO FALSE IF DON'T WANT TO RUN AGAIN
if run:
    midline, contour, background, z_midline = load_labels_safe(
        folder, label_location, filename
    )
    z = z_midline
    masks, cell_data, hcr_data = process_layer(
        z,
        folder,
        filename,
        label_location,
        results_path=results_path,
        channel_names=channel_names,
        channel_types=channel_types,
        hcr_params=[params, params, None, None],
    )
plot_all(folder, filename, label_location, results_path)