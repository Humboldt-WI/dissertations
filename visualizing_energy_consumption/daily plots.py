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

sunday_mm = week1_mm.loc['2019-03-03']
monday_mm = week1_mm.loc['2019-03-04']
tuesday_mm = week1_mm.loc['2019-03-05']
wednesday_mm = week1_mm.loc['2019-03-06']
thursday_mm = week1_mm.loc['2019-03-07']
friday_mm = week1_mm.loc['2019-03-08']
saturday_mm = week1_mm.loc['2019-03-09']


c1 = sunday_mm.MicroMoments.value_counts().sort_index()
c2 = monday_mm.MicroMoments.value_counts().sort_index()
c3 = tuesday_mm.MicroMoments.value_counts().sort_index()
c4 = wednesday_mm.MicroMoments.value_counts().sort_index()
c5 = thursday_mm.MicroMoments.value_counts().sort_index()
c6 = friday_mm.MicroMoments.value_counts().sort_index()
c7 = saturday_mm.MicroMoments.value_counts().sort_index()


c6 = pd.Series([992, 98, 99, 0, 1754])
c6.index = [0, 1, 2, 3, 4]
c7 = pd.Series([942, 87, 87, 0, 1495])
c7.index = [0, 1, 2, 3, 4]

su_counts = pd.DataFrame({'MM': my_xticks, 'No': c1})
mo_counts = pd.DataFrame({'MM': my_xticks, 'No': c2})
tue_counts = pd.DataFrame({'MM': my_xticks, 'No': c3})
we_counts = pd.DataFrame({'MM': my_xticks, 'No': c4})
thu_counts = pd.DataFrame({'MM': my_xticks, 'No': c5})
fr_counts = pd.DataFrame({'MM': my_xticks, 'No': c6})
sa_counts = pd.DataFrame({'MM': my_xticks, 'No': c7})

x = np.arange(len(my_xticks))  # the label locations
width = 0.35  # the width of the bars
w = width

fig, ax = plt.subplots()
# r1 = ax.bar(x - width, su_counts.No, w, label='Sunday')
# r2 = ax.bar(x - width*0.66, mo_counts.No, w, label='Monday')
# r3 = ax.bar(x - width*0.33, tue_counts.No, w, label='Tuesday')
# r4 = ax.bar(x, we_counts.No, w, label='Wednesday')
# r5 = ax.bar(x + width*0.33, thu_counts.No, w, label='Thursday')
# r6 = ax.bar(x + width*0.66, fr_counts.No, w, label='Friday')
r7 = ax.bar(x, sa_counts.No, w, label='Today, Saturday, 9th', color=rosa)


ax.set_ylabel('No. of Instances')
ax.set_title('Micro-Moment Instances per day in week 1')
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