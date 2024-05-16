from hcrp.labelling import get_spline
import numpy as np
from skimage.io import imread

def test_get_spline():
    stack = np.array(imread('TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant.tif'))
    layer = 26
    img = stack[layer, :, :, 3]