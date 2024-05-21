from cellpose.models import Cellpose
from cellpose import plot
import numpy as np
from skimage.measure import regionprops
from skimage.io import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector
import cv2
import os
import datetime
import pandas as pd
import bigfish.detection as detection


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


def get_midline(stack, z, name, window_name=None):
    """Manually Select ROI midline Points From Image."""
    image = stack[z]
    image_with_roi = image.copy()
    img8bit = (image_with_roi / 256).astype(np.uint8)
    image_with_roi_equalized = cv2.equalizeHist(img8bit)
    image_with_roi_equalized = np.dstack([image_with_roi_equalized] * 3)
    current_image = image_with_roi_equalized.copy()
    roi_points = []
    if window_name is None:
        window_name = f"Manually input midline for {name}"
    LINE_COLOR = (255,0,0)
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))
            cv2.circle(current_image, (x, y), 3, LINE_COLOR, -1)
            if len(roi_points) > 1:
                cv2.line(
                    current_image,
                    roi_points[-2],
                    roi_points[-1],
                    LINE_COLOR,
                    2,
                )
            cv2.imshow(window_name, current_image)


    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, current_image)
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if len(roi_points) > 1:
                cv2.destroyAllWindows()
                break
            else:
                print("You need to select at least two points.")
        if key == ord("r"):
            roi_points.clear()
            current_image = image_with_roi_equalized.copy()
            cv2.imshow(window_name, current_image)
        if key == ord("u"):
            z = min(z + 1, stack.shape[0] - 1)
            image = stack[z]
            cv2.destroyAllWindows()
            return get_midline(stack, z, name, window_name)
        if key == ord("d"):
            z = max(z - 1, 0)
            image = stack[z]
            cv2.destroyAllWindows()
            return get_midline(stack, z, name, window_name)
    cv2.destroyAllWindows()
    # change from image coords to spatial coords
    roi_points = np.array(roi_points)
    roi_points[:, [0, 1]] = roi_points[:, [1, 0]]
    roi_points = np.concatenate([roi_points, np.full((roi_points.shape[0], 1), z)], axis=1)
    return roi_points


def get_contour(stack, z, name):
    """Manually Select ROI midline Points for External Contour From Image."""
    roi_points = get_midline(
        stack, z, name, window_name=f"Manually input Contour for {name}"
    )
    return roi_points


def get_mean_region(image, high_contrast, name, size=50, vmax=None):
    """Allow the user to to select the location of a rectangular region of interest in an image and return the mean value of the region.
    The selector should be of a fixed size and the user should be able to move it around the image.
    """
    fig, ax = plt.subplots()
    if high_contrast is None:
        high_contrast = image
    ax.imshow(high_contrast, cmap="afmhot")
    ax.set_title(f"{name}")
    means = []
    centers = []

    def onselect(eclick, erelease):
        # x1, y1 = int(eclick.xdata), int(eclick.ydata)
        # x2, y2 = int(erelease.xdata), int(erelease.ydata)
        # region = image[y1:y2, x1:x2]
        # means.append(np.mean(region))
        # centers.append((x1 + x2) / 2, (y1 + y2) / 2)
        # print(f"Mean value of {name}: {means[-1]}")
        extent = selector.extents
        x1, x2, y1, y2 = [int(e) for e in extent]
        region = image[y1:y2, x1:x2]
        means.append(np.mean(region))
        centers.append([(x1 + x2) / 2, (y1 + y2) / 2])

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
    try:
        return means[-1], centers[-1]
    except IndexError:
        print("No region selected.")
        return get_mean_region(image, high_contrast, name, size, vmax)


def label(
    stack_path,
    out="data",
    mid_frac=0.5,
    channel_names=["brk", "dpp", "pmad", "nuclear"],
    size=50,
):
    """Label a single image."""
    assert channel_names[-1] == "nuclear", "The last channel must be nuclear."
    stack = imread(f"{stack_path}.tif")
    name = stack_path.split("/")[-1]
    mid_layer = int(mid_frac * stack.shape[0])
    nuclear = stack[mid_layer, :, :, 3]
    contour = get_contour(stack[:,:,:,-1], mid_layer, name)
    contour = np.hstack([contour])
    contour_out = f"{out}/{name}_contour.csv"
    contour_df = pd.DataFrame(contour, columns=["x", "y", "z"])
    contour_df.to_csv(contour_out, index=False)
    print(f"Contour points saved to {contour_out}.")
    midline = get_midline(stack[:,:,:,-1], contour[0,2], name)
    midline_out = f"{out}/{name}_midline.csv"
    midline_df = pd.DataFrame(midline, columns=["x", "y", "z"])
    midline_df.to_csv(midline_out, index=False)
    print(f"midline points saved to {out}/{name}_midline.csv.")
    background_out = f"{out}/{name}_background.csv"
    signal_background_out = f"{out}/{name}_signal_background.csv"
    columns = ["mean_intensity", "window_length", "x", "y", "z"]
    background_df = pd.DataFrame(columns=columns)
    signal_background_df = pd.DataFrame(columns=columns)
    for j, channel_name in enumerate(channel_names[:-1]):
        channel = stack[mid_layer, :, :, j]
        normalized_channel = (channel // 256).astype(np.uint8)
        normalized_channel = cv2.equalizeHist(normalized_channel)
        background, background_center = get_mean_region(
            channel, normalized_channel, f"{name} Background {channel_name}", size=size
        )
        signal_background, signal_background_center = get_mean_region(
            channel,
            normalized_channel,
            f"{name} Signal + Background {channel_name}",
            size=size,
        )
        background_df.loc[channel_name] = [
            background,
            size,
            *background_center,
            mid_layer,
        ]
        signal_background_df.loc[channel_name] = [
            signal_background,
            size,
            *signal_background_center,
            mid_layer,
        ]
    background_df.to_csv(background_out, index=True)
    print(f"Backgound data saved to {background_out}.")
    signal_background_df.to_csv(signal_background_out, index=True)
    print(f"Signal + Background data saved to {signal_background_out}.")
    print(f"Finished labelling {stack_path}.")


def label_folder(folder, mid_frac=0.5, channel_names=["brk", "dpp", "pMad", "nuclear"]):
    """Label all images in the folder."""
    assert os.path.exists(folder), f"{folder} does not exist."
    assert channel_names[-1] == "nuclear", "The last channel must be nuclear."
    stack_names = [
        name.split(".")[0] for name in os.listdir(folder) if name.endswith(".tif")
    ]
    current = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    data_out = f"data/{folder.split('/')[-1]}_{current}"
    os.makedirs(data_out, exist_ok=True)
    for i, name in enumerate(stack_names):
        print(f"Labelling {name} ({i + 1}/{len(stack_names)}).")
        label(f"{folder}/{name}", data_out, mid_frac, channel_names)
    print("Finished labelling all images.")
    return data_out
