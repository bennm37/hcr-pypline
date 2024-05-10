from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib.pyplot as plt
import numpy as np


def get_mean_region(image, name, size=50):
    """Allow the user to to select the location of a rectangular region of interest in an image and return the mean value of the region.
    The selector should be of a fixed size and the user should be able to move it around the image.
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Select a region of interest for {name}")
    means = []
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        region = image[y1:y2, x1:x2]
        means.append(np.mean(region))
        print(f"Mean value of {name}: {means[-1]}")

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )
    plt.show()
    return means[-1]
