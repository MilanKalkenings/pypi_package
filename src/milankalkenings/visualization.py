from . utils import largest_divisor
import matplotlib.pyplot as plt
from typing import List


def lines_multiplot(lines: List[List[float]],
                    title: str,
                    y_label: str,
                    x_label: str,
                    save_file: str,
                    multiplot_labels: List[str]):
    """
    creates multiple lines in the same subplot.

    :param lines: float representations of lines to plot
    :type lines: List[List[float]]

    :param title: figure title
    :type title: str

    :param multiplot_labels: line labels
    :type multiplot_labels: List[str]

    :param y_label: y label
    :type y_label: str

    :param x_label: x label
    :type x_label: str

    :param save_file: name of the file in which the figure is stored
    :type save_file: str
    """
    plt.figure(figsize=(4, 4))
    for i, line in enumerate(lines):
        plt.plot(range(len(line)), line, label=multiplot_labels[i])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.tight_layout()
    plt.ticklabel_format(useOffset=False)
    plt.savefig(save_file)


def lines_subplot(lines: List[List[float]],
                  title: str,
                  subplot_titles: List[str],
                  y_label: str,
                  x_label: str,
                  save_file: str):
    """
    creates multiple subplots within one figure.
    each subplot is a line plot.

    :param lines: float representations of lines to plot
    :type lines: List[List[float]]

    :param title: figure title
    :type title: str

    :param subplot_titles: line labels
    :type subplot_titles: List[List[str]]

    :param y_label: y label
    :type y_label: str

    :param x_label: x label
    :type x_label: str

    :param save_file: name of the file in which the figure is stored
    :type save_file: str
    """
    n_lines = len(lines)
    n_cols = largest_divisor(n=n_lines)
    n_rows = n_lines // n_cols
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    plt.suptitle(title)
    for i in range(n_lines):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.title(subplot_titles[i])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.plot(range(len(lines[i])), lines[i])
    plt.tight_layout()
    plt.savefig(save_file)
