from cellpose.models import Cellpose
from hcrp.segmentation import segment, default_hcr_params

dropbox_root = (
    "/Users/nicholb/Dropbox/Anqi/Intership/AI_segmentation/python_segmentation"
)
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"

def test_segment():
    stack_path = f"{dropbox_root}/{filename}"
    label_location = f"data/example"
    channel_names = ["brk", "dpp", "pmad", "nuclear"]
    channel_types = ["hcr", "hcr", "staining", "nuclear"]
    brk_params = default_hcr_params.copy()
    brk_params["dot_intensity_thresh"] = 0.05
    brk_params["sigma_blur"] = 1
    brk_params["verbose"] = True

    dpp_params = default_hcr_params.copy()
    dpp_params["dot_intensity_thresh"] = 0.03
    dpp_params["sigma_blur"] = 0.2
    dpp_params["verbose"] = True
    # dpp_params["fg_width"] = 0.7
    segment(stack_path, label_location, channel_names=channel_names, channel_types=channel_types, hcr_params=[brk_params, dpp_params, None, None])

if __name__ == "__main__":
    test_segment()