import numpy as np

def show_values(axes, orientation="vertical", space=.01):
    def _single(ax):
        if orientation == "vertical":
            for element in ax.patches:
                _x = element.get_x() + element.get_width() / 2
                _y = element.get_y() + element.get_height() + (element.get_height()*0.005)
                value = '{:.2f}'.format(element.get_height())
                ax.text(_x, _y, value, ha="center")
        elif orientation == "horizontal":
            i=1
            for element in ax.patches:
                _x = element.get_x() + float(space) + element.get_width() - float(space)*i*0.9
                _y = element.get_y() + element.get_height() - (element.get_height()*0.5)
                value = '{:.2f}'.format(element.get_width())
                ax.text(_x, _y, value, ha="left")
                i=+1

    if isinstance(axes.axes, np.ndarray):
        for idx, ax in np.ndenumerate(axes.axes):
            _single(ax)
    else:
        _single(axes)