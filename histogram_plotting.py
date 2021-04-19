# Author: Mattia Silvestri

"""
    Utility script to visualize multiple histograms in the same plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

########################################################################################################################


def make_bars(ax, title, rnd, agn, sbr):
    labels = ['Rows', 'Columns']
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    ax.bar(x - width, rnd, width, label='rnd')
    ax.bar(x, agn, width, label='agn')
    ax.bar(x + width, sbr, width, label='sbr')

    ax.set_title(title, fontweight='bold', fontsize='18')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(labelsize=14)
    ax.legend(loc='lower center', fontsize=16)

########################################################################################################################


if __name__ == '__main__':
    # This is an example on how you can make histograms for constraints violations
    sns.set_style('dark')
    plt.rcParams["figure.figsize"] = (15, 7)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    pls7 = np.asarray([[0, 0], [10.002, 10.378], [2.056, 2.030]])
    pls10 = np.asarray([[0, 0], [26.035, 26.987], [3.948, 3.852]])
    pls12 = np.asarray([[0, 0], [35089, 38.109], [8.545, 10.047]])
    make_bars(ax1, title='PLS-7', rnd=pls7[0], agn=pls7[1], sbr=pls7[2])
    make_bars(ax2, title='PLS-10', rnd=pls10[0], agn=pls10[1], sbr=pls10[2])
    make_bars(ax3, title='PLS-12', rnd=pls12[0], agn=pls12[1], sbr=pls12[2])
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.show()
