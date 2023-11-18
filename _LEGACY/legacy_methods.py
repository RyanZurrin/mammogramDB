from matplotlib import pyplot as plt
import numpy as np


@staticmethod
def view_histogram_grid(
    images,
    frame_number=None,
    ncols=None,
    nrows=None,
    cmap="gray",
    train_scores=None,
    hists=True,
    axis=True,
):
    """Plots a grid of images from a list of images. Optional image histogram
     alternative available.
    Parameters
    ----------
    images : list
        List of images to plot
    frame_number : int, optional
        (default is None)
        if specified, it will display the frame number of all the 3D images
    ncols : int
        (default is None)
        Number of columns in the grid
    cmap : str
        (default is 'gray')
        Color map to use
    train_scores : list, optional
        (default is None)
        If specified, it will display the train score of each image above the image
    hists : bool
        (default is True) if True, will display the histogram of the image.
    """
    images_copy = images.copy()
    for i in range(len(images)):
        if len(images[i].shape) == 3:
            if not hists:
                img = images[i].pixels
            else:
                img = O.Normalize.minmax(images[i].pixels)
            if frame_number is None:
                images_copy[i] = img[img.shape[0] // 2]
            else:
                images_copy[i] = img[frame_number]
        elif len(images[i].shape) == 2:
            if not hists:
                img = images[i].pixels
                images_copy[i] = img
            else:
                img = O.Normalize.minmax(images[i].pixels)
                images_copy[i] = O.Features.histogram(img)

    if not ncols:
        factors = [i for i in range(1, len(images) + 1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    if not nrows:
        nrows = int(len(images) / ncols) + int(len(images) % ncols)
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    # if axis is None turn off axis
    if not axis:
        axes.set_axis_off()

    # f, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.0, wspace=0.4)
    axes_1 = axes.flatten()[: len(images_copy)]
    axes_2 = axes.ravel()
    if train_scores is not None:
        titles = [float(score) for score in train_scores]

    axes_list = []
    counter = 0
    if not hists:
        for img, ax in zip(images_copy, axes_1.flatten()):
            if np.any(img):
                if len(img.shape) > 2 and img.shape[2] == 1:
                    img = img.squeeze()
                if train_scores is not None:
                    axes_list.append((ax, titles[counter], img))
                    counter += 1
                # ax.imshow(img, cmap=cmap)
        if train_scores is not None:
            axes_list = sorted(
                axes_list, key=lambda tuple_val: tuple_val[1], reverse=True
            )
            for tuple_val, ax in zip(axes_list, axes_1.flatten()):
                ax.set_title(tuple_val[1])
                ax.imshow(tuple_val[2], cmap=cmap)
    else:
        for idx, ax in enumerate(axes_2):
            if train_scores is not None:
                axes_list.append((ax, titles[counter], images_copy[idx]))
                counter += 1
            else:
                ax.hist(images_copy[idx])
        if train_scores is not None:
            axes_list = sorted(
                axes_list, key=lambda tuple_val: tuple_val[1], reverse=True
            )
            for tuple_val, ax in zip(axes_list, axes_1.flatten()):
                ax.set_title(tuple_val[1])
                ax.hist(tuple_val[2])
            plt.tight_layout()
