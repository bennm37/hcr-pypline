import matplotlib.pyplot as plt
from hcrp.segmentation import aggregate, get_random_cmap
import numpy as np


def plot_gradients(channel_names, channel_types, data, pixel_to_mu=None, bin_size=50):
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    colors = ["r", "g", "b", "k"]
    if pixel_to_mu is not None:
        dist = data["spline_dist"] * pixel_to_mu
        bin_size *= pixel_to_mu
        ax.set_xlabel("Distance Along the Midline (um)")
    else:
        dist = data["spline_dist"]
        ax.set_xlabel("Distance Along the Midline (px)")
    for i, (cname, ctype) in enumerate(zip(channel_names[:-1], channel_types[:-1])):
        # get color from cycle
        color = colors[i]
        if ctype == "hcr":
            unit = "count"
            bin_centers, c_mean, c_error = aggregate(
                dist, data[f"{cname}_{unit}"], bin_size, xmin=0
            )
            ax.errorbar(
                bin_centers,
                c_mean,
                yerr=c_error,
                label=cname,
                color=color,
                capsize=5,
            )
        else:
            unit = "mean_intensity"
            bin_centers, c_mean, c_error = aggregate(
                dist, data[f"{cname}_{unit}"], bin_size, xmin=0
            )
            ax1.errorbar(
                bin_centers,
                c_mean,
                yerr=c_error,
                label=cname,
                color=color,
                capsize=5,
            )
    ax.set_xlim(0, None)
    ax.set_ylabel("Count")
    ax.set(ylim=(0, None))
    ax1.set(ylim=(0, None))
    ax1.set_ylabel("Mean Intensity")
    fig.legend()
    plt.tight_layout()
    return fig, ax


def plot_channels(stack, channel_names, channel_types, spline, contour):
    fig, ax = plt.subplots(2, 2)
    contour = np.concatenate([contour, contour[:1]], axis=0)
    for i, (cname, ctype) in enumerate(zip(channel_names, channel_types)):
        channel = stack[:, :, i]
        ax[i // 2, i % 2].imshow(
            channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel)
        )
        ax[i // 2, i % 2].plot(spline[:, 1], spline[:, 0], c="b")
        ax[i // 2, i % 2].plot(contour[:, 1], contour[:, 0], c="r")
        ax[i // 2, i % 2].set_title(f"{cname} ({ctype})")


def plot_hcr_cell_projection(hcr_data, cell_data, masks, channel):
    fig, ax = plt.subplots()
    r_cmap = get_random_cmap(masks.max())
    color_masks = r_cmap(masks)
    # ax.imshow(channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel))
    ax.imshow(color_masks, alpha=0.8, interpolation="none")
    ax.scatter(
        hcr_data["y"], hcr_data["x"], c=r_cmap(hcr_data["closet_cell_label"]), s=1
    )
    ax.scatter(cell_data["y"], cell_data["x"], c=r_cmap(cell_data.index))
    return fig, ax


def plot_hcr(hcr_data, channel):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    r_cmap = get_random_cmap(hcr_data["closet_cell_label"].max())
    ax[0].imshow(channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel))
    ax[0].axis("off")
    ax[1].imshow(channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel))
    ax[1].scatter(hcr_data["y"], hcr_data["x"], s=2)
    ax[1].axis("off")
    return fig, ax


def plot_cell_property(cell_data, channel, prop_name, f=None):
    fig, ax = plt.subplots()
    ax.imshow(channel, cmap="afmhot", vmax=np.mean(channel) + 3 * np.std(channel))
    if f is not None:
        color_data = f(cell_data[prop_name])
    else:
        color_data = cell_data[prop_name]
    ax.scatter(cell_data["y"], cell_data["x"], c=color_data)
    ax.axis("off")
    ax.set_title(f"Cell Property: {prop_name}")
    return fig, ax
