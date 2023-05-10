import torch
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


def images_subplot(images: List[torch.Tensor],
                   title: str,
                   subplot_titles: List[str],
                   save_file: str):
    """
    creates multiple subplots within one figure.
    each subplot shows an image.

    :param images: torch Tensors representing RGB images
    :type images: List[torch.Tensor]

    :param title: figure title
    :type title: str

    :param subplot_titles: image labels
    :type subplot_titles: List[str]

    :param save_file: name of the file in which the figure is stored
    :type save_file: str
    """
    n_images = len(images)
    num_cols = largest_divisor(n=n_images)
    num_rows = n_images // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    plt.suptitle(title)
    if num_cols == 1:
        for i, img in enumerate(images):
            image_tensor = images[i].detach()
            image_array = image_tensor.permute(1, 2, 0).numpy()
            axs[i].imshow(image_array)
            axs[i].set_title(subplot_titles[i])
            axs[i].axis('off')
    else:
        for i, img in enumerate(images):
            row_idx = i // num_cols
            col_idx = i % num_cols
            image_tensor = images[i].detach()
            image_array = image_tensor.permute(1, 2, 0).numpy()
            axs[row_idx, col_idx].imshow(image_array)
            axs[row_idx, col_idx].set_title(subplot_titles[i])
            axs[row_idx, col_idx].axis('off')
    plt.tight_layout()
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
    :type subplot_titles: List[str]

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
        plt.ticklabel_format(useOffset=False)
        plt.plot(range(len(lines[i])), lines[i])
    plt.tight_layout()
    plt.savefig(save_file)
