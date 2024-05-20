from cellpose.models import Cellpose
from hcrp.segmentation import segment

dropbox_root = (
    "/Users/nicholb/Dropbox/Anqi/Intership/AI_segmentation/python_segmentation"
)
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"

def test_segment():
    stack_path = f"{dropbox_root}/{filename}"
    label_location = f"data/example"
    channel_names = ["brk", "dpp", "pmad", "nuclear"]
    channel_types = ["hcr", "hcr", "staining", "nuclear"]
    hcr_params={
        "sigma_blur": 1,
        "pixel_intensity_thresh": 0.0,
        "fg_width": 0.2,
        "dot_intensity_thresh": 0.04,
        "verbose": True,
    }
    segment(stack_path, label_location, channel_names=channel_names, channel_types=channel_types, hcr_params=hcr_params)

if __name__ == "__main__":
    test_segment()