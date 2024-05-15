from pymorphogen.hcr import process
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
maxima = process(image, high_contrast=normalized)
plt.imshow(image, vmax=10*np.mean(image), cmap="afmhot")
plt.scatter(maxima[:, 1], maxima[:, 0], c="r", s=1)
plt.show()