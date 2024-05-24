from hcrp import * 
import pandas as pd 
import matplotlib.pyplot as plt

cell_data = pd.read_csv("data/Stg01_Emb02_T101_results/Stg01_Emb02_T101_cell_data.csv")
z_range = cell_data["z"].unique()
channel_names = ["brk", "dpp", "pmad", "nuclear"]
channel_types = ["hcr", "hcr", "staining", "nuclear"]
plot_layer_gradients(z_range, channel_names, channel_types, cell_data, pixel_to_mu=1/2.8609, bin_size=50, err_type="std")
plt.show()