
from cellpose.models import Cellpose
from hcrp.segmentation import segment, default_hcr_params
from hcrp.labelling import label_folder, load_labels
from hcrp.plotting import plot_gradients, plot_channels
from skimage.io import imread
import matplotlib.pyplot as plt
import os

dropbox = "/Users/nicholb/Dropbox"
folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"

# label(f"{dropbox_root}/{filename}", "data/example", 0.5)

def segment_folder(folder=folder, label_location="data/example"):
    stack_names = [
        name.split(".")[0] for name in os.listdir(folder) if name.endswith(".tif")
    ]
    channel_names = ["brk", "dpp", "pmad", "nuclear"]
    channel_types = ["hcr", "hcr", "staining", "nuclear"]
    brk_params = default_hcr_params.copy()
    brk_params["dot_intensity_thresh"] = 0.03
    brk_params["sigma_blur"] = 1
    brk_params["verbose"] = False

    dpp_params = default_hcr_params.copy()
    dpp_params["dot_intensity_thresh"] = 0.03
    dpp_params["sigma_blur"] = 0.2
    dpp_params["verbose"] = False
    for stack_name in stack_names:
        contour, midline, background, z = load_labels(label_location, stack_name)
        # dpp_params["fg_width"] = 0.7
        masks, data = segment(f"{folder}/{stack_name}", label_location, channel_names=channel_names, channel_types=channel_types, hcr_params=[brk_params, dpp_params, None, None], verbose=False)
        stack = imread(f"{folder}/{stack_name}.tif")
        plot_channels(stack[z], channel_names, channel_types, contour, midline)
        plot_gradients(channel_names, channel_types, data)
        plt.show()


if __name__ == "__main__":
    label_location = "data/Limb_Ext_Stg01"
    if not os.path.exists(label_location):
        label_location = label_folder(folder)
    segment_folder(folder, label_location)