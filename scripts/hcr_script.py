from pymorphogen.hcr import process_2
from skimage.io import imread, imsave
import cv2
import matplotlib.pyplot as plt
import numpy as np 

read_stack = False
if read_stack:
    stack = imread("data/TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant.tif")
    image = stack[26, :, :,]
    # write the image
    imsave("data/seperated.tif", image) 
else:
    image = imread("data/seperated.tif")
    image = (image//256).astype(np.uint8)
    normalized = cv2.equalizeHist(image)

# # vmax = 3 * np.mean(image)
labels, props = process_2(image, high_contrast=normalized, sigma_blur=0.2, pixel_intensity_thresh=0.000, fg_width=0.05, dot_intensity_thresh=0.1, size=50)
# labels, props = process_2(image, high_contrast=normalized, sigma_blur=0.5, pixel_intensity_thresh=0.000, fg_width=0.05, dot_intensity_thresh=0.1, size=50)
centroids = np.array([prop.centroid for prop in props])
intensities = np.array([prop.mean_intensity for prop in props])
# plt.imshow(image, vmax=10*np.mean(image), cmap="afmhot")
# plt.scatter(centroids[:, 1], centroids[:, 0], c="green", s=1)
# plt.show()
n_centroids = centroids.shape[0]
p = 1. * np.arange(n_centroids) / (n_centroids - 1)
fig, ax = plt.subplots()
ax.plot(np.sort(centroids[:,0]), p, c="r")
ax.set(xlabel="Centroid Location", ylabel="CDF")
ax2 = ax.twinx()
ax2.hist(centroids[:,0], bins=20, density=True, alpha=0.5)
ax2.set(ylabel="Density")
plt.show()
# plt.imshow(image, vmax=10*np.mean(image), cmap="afmhot")
# plt.scatter(maxima[:, 1], maxima[:, 0], c="r", s=1)
# plt.show()