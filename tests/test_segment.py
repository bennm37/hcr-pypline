from cellpose.models import Cellpose
from hcrp.segmentation import segment, default_hcr_params, aggregate
from hcrp.labelling import label
import matplotlib.pyplot as plt

dropbox_root = (
    "/Users/nicholb/Dropbox/Anqi/Intership/AI_segmentation/python_segmentation"
)
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"
# label(f"{dropbox_root}/{filename}", "data/example", 0.5)
def test_segment():
    stack_path = f"{dropbox_root}/{filename}"
    label_location = f"data/example"
    channel_names = ["brk", "dpp", "pmad", "nuclear"]
    channel_types = ["hcr", "hcr", "staining", "nuclear"]
    brk_params = default_hcr_params.copy()
    brk_params["dot_intensity_thresh"] = 0.03
    brk_params["sigma_blur"] = 1
    brk_params["verbose"] = True

    dpp_params = default_hcr_params.copy()
    dpp_params["dot_intensity_thresh"] = 0.03
    dpp_params["sigma_blur"] = 0.2
    dpp_params["verbose"] = True
    # dpp_params["fg_width"] = 0.7
    masks, data = segment(stack_path, label_location, channel_names=channel_names, channel_types=channel_types, hcr_params=[brk_params, dpp_params, None, None])
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
    plt.show()

if __name__ == "__main__":
    test_segment()