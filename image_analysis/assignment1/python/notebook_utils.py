import matplotlib.pyplot as plt
from libtiff import TIFF


def display_grayscale_images(paths, titles=None, figsize=None):
    assert len(paths) > 0
    assert all(path.endswith(".tif") for path in paths)

    titles = titles or paths
    images = [TIFF.open(path).read_image() for path in paths]

    fig, ax = plt.subplots(1, len(images), figsize=figsize, squeeze=False)

    for i, (title, image) in enumerate(zip(titles, images)):
        ax[0][i].imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        ax[0][i].axis("off")
        ax[0][i].set_title(title)


def display_rgb_images(paths, titles=None, figsize=None):
    assert len(paths) > 0
    assert all(path.endswith(".tif") for path in paths)

    titles = titles or paths
    images = [TIFF.open(path).read_image() for path in paths]

    fig, ax = plt.subplots(1, len(images), figsize=figsize, squeeze=False)

    for i, (title, image) in enumerate(zip(titles, images)):
        ax[0][i].imshow(image, interpolation="nearest", vmin=0, vmax=255)
        ax[0][i].axis("off")
        ax[0][i].set_title(title)
