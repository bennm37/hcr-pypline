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
# label_folder(folder, mid_frac=0.5)

# backgrounds = [800, 1500, 800, 500]
backgrounds = [800, 1500, 800, 500]
for i, file in enumerate(files):
    print(f"Processing {file}")
    rna = stack.read_image(f"{folder}/{file}").astype(np.float32)
    px_nano = 1000 / 2.8906
    pz_nano = 1000
    voxel_size = (pz_nano, px_nano, px_nano)
    spot_radius = (700, 350, 350)
    c = 1
    z = 15
    rna_z = rna[z, :, :, c]
    masks, centroids = quantify_hcr(rna_z, backgrounds[i], 0.2, dot_intensity_thresh=0.0275)
    centroids_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
        image=rna[z, :, :, c].astype(np.uint16),
        spots=centroids,
        voxel_size=(103, 103),
        spot_radius=(150, 150),
        alpha=0.90,  # alpha impacts the number of spots per candidate region
        beta=1,  # beta impacts the number of candidate regions to decompose
        gamma=5,
    )  # gamma the filtering step to denoise the image
    print("detected centroids before decomposition")
    print("\r shape: {0}".format(centroids.shape))
    print("\r dtype: {0}".format(centroids.dtype), "\n")
    print("detected centroids after decomposition")
    print("\r shape: {0}".format(centroids_post_decomposition.shape))
    print("\r dtype: {0}".format(centroids_post_decomposition.dtype))
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.set_size_inches(10, 7)
    ax[0].imshow(rna_z, cmap="afmhot", vmax=np.mean(rna_z) + 5 * np.std(rna_z))
    # ax[0].scatter(spots_z[:, 1], spots_z[:, 0], alpha=0.5, c="b")
    ax[0].scatter(centroids_post_decomposition[:, 1], centroids_post_decomposition[:, 0], alpha=0.5, c="y")
    ax[0].axis("off")
    ax[1].scatter(centroids[:, 1], centroids[:, 0], alpha=0.5, c="g")
    ax[1].imshow(rna_z, cmap="afmhot", vmax=np.mean(rna_z) + 5 * np.std(rna_z))
    ax[1].axis("off")
    plt.show()
