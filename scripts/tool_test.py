import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolToggleBase
import matplotlib as mpl
mpl.use('TkAgg')

class toggle_example(ToolToggleBase):
    default_keymap = 't'
    description = 'Toggle Button'

t= np.arange(1e5)/1e5
y= np.random.randn(int(1e5))
    
plt.close(fig=1)
plt.figure(1)
plt.plot(t,y)

plt.gcf().canvas.manager.toolmanager.add_tool('Toggle example', toggle_example)
plt.gcf().canvas.manager.toolbar.add_tool('Toggle example', 'zoompan', 2)
plt.ioff()
plt.show()
