import matplotlib.pyplot as plt
from cellpose.models import Cellpose

model = Cellpose(gpu=True, model_type='cyto')