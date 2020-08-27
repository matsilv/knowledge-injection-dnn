# @Author: Mattia Silvestri

"""
    Utility script to visualize multiple histograms in the same plot.
"""

import seaborn as sns
# Set default seaborn style
sns.set()
import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["figure.figsize"] = (12, 5)

########################################################################################################################


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#######################################################################################################################

# Set the values for each method in the following lists
#random = [13392, 12111]
random = [30753]
sbr = [27749]
msk = [21863]
agnostic = [23321]
#model_agnostic = [11696, 10299]

# Set y values limit
plt.ylim(0, 31000)

# Set x labels
#labels = ['Rows constraint\nviolations', 'Columns constraint\nviolations']
labels = ['Columns constraints violations']

# Labels locations
x = np.arange(len(labels))
# Width of the bars
width = 0.01

# Set Seaborn style
sns.set_style("dark")

# Create subplots
fig, ax = plt.subplots()

# Create rectangles with eventually many bars
rects3 = ax.bar(x - 5 * width, random, width, label="rnd")
rects1 = ax.bar(x - 4 * width, agnostic, width, label='agn')
rects2 = ax.bar(x - 3 * width, sbr, width, label='sbr')
rects3 = ax.bar(x - 2 * width, msk, width, label="mask")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('# of columns costraints violations on 500 solutions generation', fontsize="small")
#ax.set_title('Constraint Violations')
#ax.set_xticks([-0.25, 0.7])
ax.set_xticks([0])
ax.set_xticklabels(labels)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize="small")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize="small", frameon=False)

'''autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)'''


#fig.tight_layout()

#plt.show(ax)
plt.savefig("Figura 12 - violazioni del vincolo di colonna nella generazione di 500 soluzioni - (1k solutions pool).png")