import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes as ax
import numpy as np
import seaborn as sns
from mpl_toolkits import axisartist

qud = pd.read_csv("QUD_app-1 (desktop).csv", index_col=0, parse_dates=True)
# qud.info()
my_xticks = ["good usage", "turn on", "turn off",
             "excessive\n power\n consumption", "consumption\n while outside"]


march_mm = qud.loc['2019-03-03':'2019-03-31']
may_mm = qud.loc['2019-05-01':'2019-05-31']
# march_mm.info()
# may_mm.info()

counts = march_mm.MicroMoments.value_counts().sort_index()
countsmay = may_mm.MicroMoments.value_counts().sort_index()
z = pd.Series([0])
countsmay = countsmay.append(z, ignore_index=True)
print(countsmay)
march_counts = pd.DataFrame({'MM': my_xticks,
                            'No': counts})
may_counts = pd.DataFrame({'MM': my_xticks,
                            'No': countsmay})
# plt.bar(march_counts.MM, march_counts.No)
print(may_counts)
x = np.arange(len(my_xticks))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, march_counts.No, width, label='March', color='#800080')
rects2 = ax.bar(x + width/2, may_counts.No, width, label='May', color='#008000')

ax.set_ylabel('No. of Instances')
ax.set_title('Micro-Moment Instances per Month')
ax.set_xticks(x)
ax.set_xticklabels(my_xticks)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
