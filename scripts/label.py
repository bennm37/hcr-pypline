from hcrp.labelling import label, label_folder
from hcrp import get_path
folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset4_dad_dpp_pMad_Nuclei/"
filename = "Td_Hoechst_pMad488_dpp546_dad647_20240522_Embryo02_T1.tif"
label_location = "data/SilverFish_Pilot"
# LABEL 1 STACK
label(f"{folder}/{filename}", label_location)
## LABEL A WHOLE FOLDER
# label_folder(folder)
