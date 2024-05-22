from hcrp.labelling import label, label_folder
from hcrp import get_path

folder = f"{get_path('dropbox.txt')}/Anqi/Intership/AI_segmentation/Dataset1_brk_dpp_pMad_Nuclei/Limb_Ext_Stg01"
# label 1 stack
# label(f"{folder}/Stg01_Emb01_Ant01")
# label a whole folder
label_folder(folder)
