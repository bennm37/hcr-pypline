from cellpose.models import Cellpose
from skimage import io
import matplotlib.pyplot as plt
import numpy as np  
from hcrp.segmentation import get_cells, segment

def test_cellpose():
    # for cytoplasm
    dropbox = "/Users/nicholb/Dropbox"
    folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
    stack = np.array(io.imread(f"{folder}/Stg01_Emb02_Lb01.tif"))
    layer = 10
    img = stack[layer, :, :, 3]
    pmad = stack[layer, :, :, 2]
    model = Cellpose(gpu=True, model_type='cyto')
    masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0,0])
    plt.imshow(masks)
    plt.show()

def test_get_cells():
    dropbox = "/Users/nicholb/Dropbox"
    folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
    stack = np.array(io.imread(f"{folder}/Stg01_Emb02_Lb01.tif"))
    layer = 10
    img = stack[layer, :, :, 3]
    masks, props = get_cells(img)
    plt.imshow(masks)
    plt.show()

def test_segment():
    dropbox = "/Users/nicholb/Dropbox"
    folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
    segment(f"{folder}/Stg01_Emb02_Lb01", f"data/Limb_Ext_Stg01")

if __name__ == "__main__":
    test_cellpose()
    test_get_cells()
    test_segment()