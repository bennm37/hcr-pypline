from hcrp.labelling import get_midline, get_contour, get_mean_region, label
from hcrp import get_path
import numpy as np
from skimage.io import imread

folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"


def test_get_midline():
    stack = np.array(imread(f"{folder}/{filename}.tif"))
    layer = 26
    img = stack[layer, :, :, 3]
    get_midline(img, f"{filename}")


def test_get_contour():
    stack = np.array(imread(f"{folder}/{filename}.tif"))
    layer = 26
    img = stack[layer, :, :, 3]
    get_contour(img, f"{filename}")


def test_get_mean_region():
    stack = np.array(
        imread(
            f"{folder}/{filename}.tif",
        )
    )
    layer = 26
    img = stack[layer, :, :, 3]
    get_mean_region(img, None, "Background", size=50, vmax=None)


def test_label():
    stack_path = (
        "data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"
    )
    label(stack_path, out="data/example")


if __name__ == "__main__":
    # test_get_midline()
    # test_get_contour()
    # test_get_mean_region()
    test_label()
