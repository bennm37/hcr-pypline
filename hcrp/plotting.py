import matplotlib.pyplot as plt
from hcrp.segmentation import aggregate
import numpy as np

def plot_gradients(channel_names, channel_types, data):
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    colors = ["r", "g", "b", "k"]
    xshift = 0
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
            # get color from cycle
        color = colors[i]
        if ctype == "hcr":
            unit = "count"
            bin_centers, c_mean, c_error = aggregate(data["spline_dist"], data[f"{cname}_{unit}"], 50)
            ax.errorbar(bin_centers + xshift * i, c_mean, yerr=c_error, label=cname, color=color, capsize=5)
        else:
            unit = "mean_intensity"
            bin_centers, c_mean, c_error = aggregate(data["spline_dist"], data[f"{cname}_{unit}"], 50)
            ax1.errorbar(bin_centers + xshift * i, c_mean, yerr=c_error, label=cname, color=color, capsize=5)
    ax.set_xlabel("Distance Along the Midline (px)")
    ax.set_ylabel("Count")
    ax.set(ylim=(0, None))
    ax1.set(ylim=(0, None))
    ax1.set_ylabel("Mean Intensity")
    fig.legend()


def plot_channels(stack, channel_names, channel_types, spline, contour):
    fig, ax = plt.subplots(2, 2)
    contour = np.concatenate([contour, contour[:1]], axis=0)
    for i, (cname, ctype) in enumerate(zip(channel_names, channel_types)):
        channel = stack[:, :, i]
        ax[i//2, i%2].imshow(channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel))
        ax[i//2, i%2].plot(spline[:, 1], spline[:, 0], c="b")
        ax[i//2, i%2].plot(contour[:, 1], contour[:, 0], c="r")
        ax[i//2, i%2].set_title(f"{cname} ({ctype})")
