import numpy as np
from cellpose.models import Cellpose

model = Cellpose(gpu=True, model_type="cyto")
