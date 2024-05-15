import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import graph

# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2, x3, y3 = 28, 28, 44, 52, 60, 28
r1, r2, r3 = 16, 20, 14
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3**2
image = np.logical_or(mask_circle1, mask_circle2)
image = np.logical_or(image, mask_circle3)
plt.imshow(image)
plt.show()
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)
rag = graph.rag_boundary(labels, image.astype(float))
graph.show_rag(labels, rag, np.dstack([labels]*3), img_cmap="YlOrBr", edge_cmap="coolwarm", edge_width=1)
plt.show()