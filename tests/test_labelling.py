from hcrp.labelling import get_midline, get_contour, get_mean_region, label
import numpy as np
from skimage.io import imread

# dropbox_root = "/Users/huanga/The Francis Crick Dropbox/VincentJ/Anqi/Intership/AI_segmentation/python_segmentation/"
dropbox_root = (
    "/Users/nicholb/Dropbox/Anqi/Intership/AI_segmentation/python_segmentation"
)
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"


def test_get_midline():
    stack = np.array(imread(f"{dropbox_root}/{filename}.tif"))
    layer = 26
    img = stack[layer, :, :, 3]
    get_midline(img, f"{filename}")


def test_get_contour():
    stack = np.array(imread(f"{dropbox_root}/{filename}.tif"))
    layer = 26
    img = stack[layer, :, :, 3]
    get_contour(img, f"{filename}")


def test_get_mean_region():
    stack = np.array(
        imread(
            f"{dropbox_root}/{filename}.tif",
        )
    )
    layer = 26
    img = stack[layer, :, :, 3]
    get_mean_region(img, None, "Background", size=50, vmax=None)

def test_label():
    stack_path = "data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"
    label(stack_path)

if __name__ == "__main__":
    # test_get_midline()
    # test_get_contour()
    # test_get_mean_region()
    test_label()
