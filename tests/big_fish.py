import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.plot as plot
import matplotlib.pyplot as plt
import numpy as np
from hcrp.hcr import quantify_hcr
from hcrp.labelling import label_folder

dropbox = "/Users/nicholb/Dropbox"
folder = f"{dropbox}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
files = [
    "Stg01_Emb01_Ant01.tif",
    "Stg01_Emb01_T102.tif",
    "Stg01_Emb01_Ant01.tif",
    "Stg01_Emb01_Ant02.tif",
    "Stg01_Emb01_Lb01.tif",
    "Stg01_Emb01_Lb02.tif",
    "Stg01_Emb01_Mx01.tif",
    "Stg01_Emb01_Mx02.tif",
    "Stg01_Emb01_T101.tif",
]
# label_folder(folder, mid_frac=0.3)

backgrounds = [800, 1500, 800, 500]
for i, file in enumerate(files):
    print(f"Processing {file}")
    rna = stack.read_image(f"{folder}/{file}").astype(np.float32)
    px_nano = 1000 / 2.8906
    pz_nano = 1000
    c = 1
    spots, threshold = detection.detect_spots(
        images=rna[:, :, :, c],
        # threshold=500,
        return_threshold=True,
        voxel_size=(pz_nano, px_nano, px_nano),
        spot_radius=(700,350,350),
    )
    print("detected spots")
    print("\r shape: {0}".format(spots.shape))
    print("\r dtype: {0}".format(spots.dtype))
    print("\r threshold: {0}".format(threshold))
    z = 15
    spots_z = spots[spots[:, 0] == z, 1:]
    rna_z = rna[z, :, :, c]
    masks, centroids = quantify_hcr(rna_z, backgrounds[i], 0.2, dot_intensity_thresh=0.03)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    ax[0].imshow(rna_z, cmap="afmhot", vmax=np.mean(spots_z) + 20 * np.std(spots_z))
    ax[0].scatter(spots_z[:, 1], spots_z[:, 0], alpha=0.5, c="b")
    ax[0].scatter(centroids[:, 1], centroids[:, 0], alpha=0.5, c="g")
    ax[0].axis("off")
    ax[1].imshow(rna_z, cmap="afmhot", vmax=np.mean(spots_z) + 20 * np.std(spots_z))
    ax[1].axis("off")
    plt.show()
