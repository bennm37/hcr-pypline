from hcrp.labelling import label, label_folder
from hcrp import get_path

folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
filename = "Stg01_Emb03_T102.tif"
data_out = "data/Limb_Ext_Stg01_test"
# LABEL 1 STACK
label(f"{folder}/{filename}", data_out)
## LABEL A WHOLE FOLDER
# label_folder(folder)
