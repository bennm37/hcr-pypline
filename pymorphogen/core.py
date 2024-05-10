from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase, ToolToggleBase
plt.rcParams['toolbar'] = 'toolmanager'


class ZStack:
    def __init__(self, data, name='ZStack Viewer'):
        self.data = data
        self.current_slice = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(name)
        # plt.subplots_adjust(left=0.1, bottom=0.25)
        self.slider_ax = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(self.slider_ax, 'z', 0, data.shape[2] - 1, valinit=self.current_slice, valstep=1)
        self.slider.on_changed(self.update)
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # add drawing tools to the figure window
        self.fig.canvas.manager.toolbar.add_tool('list', ListTools)

        self.update(0)

    def update(self, val):
        self.current_slice = int(self.slider.val)
        self.ax.imshow(self.data[:, :, self.current_slice, :], cmap='gray')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'right':
            self.slider.set_val((self.current_slice + 1) % self.data.shape[2])
        elif event.key == 'left':
            self.slider.set_val((self.current_slice - 1) % self.data.shape[2])

    def show(self):
        # plt.tight_layout()
        plt.show()

class ListTools(ToolBase):
    """List all the tools controlled by the `ToolManager`."""
    default_keymap = 'm'  # keyboard shortcut
    description = 'List Tools'

    def trigger(self, *args, **kwargs):
        print('_' * 80)
        fmt_tool = "{:12} {:45} {}".format
        print(fmt_tool('Name (id)', 'Tool description', 'Keymap'))
        print('-' * 80)
        tools = self.toolmanager.tools
        for name in sorted(tools):
            if not tools[name].description:
                continue
            keys = ', '.join(sorted(self.toolmanager.get_tool_keymap(name)))
            print(fmt_tool(name, tools[name].description, keys))
        print('_' * 80)
        fmt_active_toggle = "{!s:12} {!s:45}".format
        print("Active Toggle tools")
        print(fmt_active_toggle("Group", "Active"))
        print('-' * 80)
        for group, active in self.toolmanager.active_toggle.items():
            print(fmt_active_toggle(group, active))