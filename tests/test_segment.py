from hcrp.segmentation import segment

dropbox_root = (
    "/Users/nicholb/Dropbox/Anqi/Intership/AI_segmentation/python_segmentation"
)
filename = "TdEmbryo_Hoechst_pMad488_dpp546_brk647_20240506_LimbExtension10-Ant"

def test_segment():
    stack_path = f"{dropbox_root}/{filename}"
    label_location = f"{dropbox_root}/data"
    segment(stack_path, label_location)
    assert True 