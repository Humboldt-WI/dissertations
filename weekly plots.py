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

week1_mm = qud.loc['2019-03-03':'2019-03-09']
week2_mm = qud.loc['2019-03-10':'2019-03-16']
week3_mm = qud.loc['2019-03-17':'2019-03-21']  #
week4_mm = qud.loc['2019-05-05':'2019-05-11']
week5_mm = qud.loc['2019-05-12':'2019-05-18']
week6_mm = qud.loc['2019-05-19':'2019-05-26']

# march_mm.info()
# may_mm.info()

c1 = week1_mm.MicroMoments.value_counts().sort_index()
c2 = week2_mm.MicroMoments.value_counts().sort_index()
c3 = week3_mm.MicroMoments.value_counts().sort_index()  #
c4 = week4_mm.MicroMoments.value_counts().sort_index()
c5 = week5_mm.MicroMoments.value_counts().sort_index()
c6 = week6_mm.MicroMoments.value_counts().sort_index()

z = pd.Series([0])
c4 = c4.append(z, ignore_index=True)
c5 = c5.append(z, ignore_index=True)
c6 = c6.append(z, ignore_index=True)

w1_counts = pd.DataFrame({'MM': my_xticks, 'No': c1})
w2_counts = pd.DataFrame({'MM': my_xticks, 'No': c2})
w3_counts = pd.DataFrame({'MM': my_xticks, 'No': c3})  #
w4_counts = pd.DataFrame({'MM': my_xticks, 'No': c4})
w5_counts = pd.DataFrame({'MM': my_xticks, 'No': c5})
w6_counts = pd.DataFrame({'MM': my_xticks, 'No': c6})

# plt.bar(march_counts.MM, march_counts.No)
x = np.arange(len(my_xticks))  # the label locations
width = 0.35  # the width of the bars
w = width/5

fig, ax = plt.subplots()
r1 = ax.bar(x - width*1.2, w1_counts.No, w, label='week 1')
r2 = ax.bar(x - width*0.8, w2_counts.No, w, label='week 2')
r3 = ax.bar(x - width*0.4, w3_counts.No, w, label='half week 3')
r4 = ax.bar(x, w4_counts.No, w, label='week 4')
r5 = ax.bar(x + width*0.4, w5_counts.No, w, label='week 5')
r6 = ax.bar(x + width*0.8, w6_counts.No, w, label='week 6')


ax.set_ylabel('No. of Instances')
ax.set_title('Micro-Moment Instances per week')
ax.set_xticks(x)
ax.set_xticklabels(my_xticks)
ax.legend()


'''ax.bar_label(r1, padding=3)
ax.bar_label(r2, padding=3)
ax.bar_label(r3, padding=3)
ax.bar_label(r4, padding=3)
ax.bar_label(r5, padding=3)
ax.bar_label(r6, padding=3)'''

fig.tight_layout()

plt.show()