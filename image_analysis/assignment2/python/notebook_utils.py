import matplotlib.pyplot as plt
import numpy as np
from libtiff import TIFF


def display_images(
    paths_or_images,
    titles=None,
    subplot_shape=None,
    figsize=None,
):
    assert len(paths_or_images) > 0
    assert titles is None or len(titles) == len(paths_or_images)
    assert set(map(type, paths_or_images)) in [{str}, {np.ndarray}]

    if isinstance(paths_or_images[0], str):
        assert all(path.endswith(".tif") for path in paths_or_images)
        images = [TIFF.open(path).read_image() for path in paths_or_images]
        titles = titles or paths_or_images

    else:
        images = paths_or_images
        titles = titles or [f"Image {i}" for i in range(len(images))]

    if subplot_shape is None:
        subplot_shape = (1, len(images))

    fig, ax = plt.subplots(*subplot_shape, figsize=figsize, squeeze=False)
    ax = ax.flatten()
    for i, (title, image) in enumerate(zip(titles, images)):
        # convert complex images to float
        if image.dtype == np.complex128:
            image = np.log(np.abs(image) + 1)
            image = image / image.max()
            image = (image * 255).astype(np.uint8)

        # if the image is grayscale, use the "gray" colormap
        cmap = "gray" if len(image.shape) == 2 else None

        ax[i].imshow(image, cmap=cmap, interpolation="nearest", vmin=0, vmax=255)
        ax[i].axis("off")
        ax[i].set_title(title)
