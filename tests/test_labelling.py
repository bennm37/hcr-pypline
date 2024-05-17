from hcrp.labelling import get_spline, get_contour
import numpy as np
from skimage.io import imread

def test_get_spline():
    stack = np.array(imread('data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant.tif'))
    layer = 26
    img = stack[layer, :, :, 3]
    get_spline(img, 'test_spline')

def test_get_contour():
    stack = np.array(imread('data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant.tif'))
    layer = 26
    img = stack[layer, :, :, 3]
    get_contour(img, 'test_contour')


if __name__ == '__main__':
    # test_get_spline()
    test_get_contour()