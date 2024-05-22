from hcrp import get_path
from hcrp.labelling import get_z_range, label_folder
import os
from skimage.io import imread
import pandas as pd
import numpy as np


def get_z_ranges(folder, label_location, data_out=None):
    stack_names = [
        name.split(".")[0] for name in os.listdir(folder) if name.endswith(".tif")
    ]
    if data_out is None:
        data_out = f"{label_location}_z_ranges"
    if not os.path.exists(label_location):
        label_location = label_folder(folder)
    os.mkdir(data_out)
    for stack_name in stack_names:
        stack = imread(f"{folder}/{stack_name}.tif")
        z_range = get_z_range(stack[:, :, :, -1], label_location, stack_name)
        df = pd.DataFrame(np.array([z_range]), columns=["z_low", "z_high"])
        df.to_csv(f"{label_location}_z_ranges/{stack_name}_z_range.csv", index=False)


if __name__ == "__main__":
    label_location = "data/Limb_Ext_Stg01"
    folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
    get_z_ranges(folder, label_location)
