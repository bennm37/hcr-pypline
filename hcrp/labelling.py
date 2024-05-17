from cellpose.models import Cellpose
from cellpose import plot
import numpy as np
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector
import cv2


class FixedSizeRectangleSelector(RectangleSelector):
    def __init__(self, ax, onselect, size=50, **kwargs):
        super().__init__(ax, onselect, **kwargs)
        self.extents = 0, size, 0, size
        self._allow_creation = False
        self.update()

    def _onmove(self, event):
        """
        Motion notify event handler.

        This class only allows:
        - Translate
        """
        eventpress = self._eventpress
        # The calculations are done for rotation at zero: we apply inverse
        # transformation to events except when we rotate and move
        move = self._active_handle == "C"

        xdata, ydata = self._get_data_coords(event)
        dx = xdata - eventpress.xdata
        dy = ydata - eventpress.ydata
        x0, x1, y0, y1 = self._extents_on_press
        if move:
            x0, x1, y0, y1 = self._extents_on_press
            dx = xdata - eventpress.xdata
            dy = ydata - eventpress.ydata
            x0 += dx
            x1 += dx
            y0 += dy
            y1 += dy
        self.extents = x0, x1, y0, y1


def get_spline(image, name):
    """Manually Select ROI Spline Points From Image."""
    image_with_roi = image.copy()
    img8bit = (image_with_roi / 256).astype(np.uint8)
    image_with_roi_equalized = cv2.equalizeHist(img8bit)
    roi_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(image_with_roi_equalized, (x, y), 3, (0, 0, 255), -1)
            if len(roi_points) > 1:
                cv2.line(
                    image_with_roi_equalized,
                    roi_points[-2],
                    roi_points[-1],
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("Select ROI", image_with_roi_equalized)

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", image_with_roi_equalized)
    cv2.setMouseCallback("Select ROI", mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    csv_filename = f"data/{name}.csv"
    np.savetxt(csv_filename, roi_points, delimiter=",", fmt="%d")
    print(f"ROI points saved to {csv_filename}.")
    return roi_points


def get_contour(image, name):
    """Manually Select ROI Spline Points for External Contour From Image."""
    roi_points = get_spline(image, name)
    return roi_points


def get_mean_region(image, high_contrast, name, size=50, vmax=None):
    """Allow the user to to select the location of a rectangular region of interest in an image and return the mean value of the region.
    The selector should be of a fixed size and the user should be able to move it around the image.
    """
    fig, ax = plt.subplots()
    if high_contrast is None:
        high_contrast = image
    ax.imshow(high_contrast, cmap="afmhot")
    ax.set_title(f"Select a region of interest for {name}")
    means = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        region = image[y1:y2, x1:x2]
        means.append(np.mean(region))
        print(f"Mean value of {name}: {means[-1]}")

    selector = FixedSizeRectangleSelector(
        ax,
        onselect,
        size=size,
        useblit=True,
        button=[1, 3],  # disable middle button
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )

    def reset(event):
        if event.key == "r":
            means.clear()
            selector.extents = 0, size, 0, size
            selector.update()
            ax.set_title(f"Select a region of interest for {name}")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", reset)
    plt.show()
    return means[-1]
