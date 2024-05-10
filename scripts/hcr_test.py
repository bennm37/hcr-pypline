from pymorphogen.hcr import get_mean_region
import numpy as np

image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
background = get_mean_region(image, 'Background', size=10)
signal_background = get_mean_region(image, 'Signal + Background', size=10)
print(f"Background: {background}")
print(f"Signal + Background: {signal_background}")