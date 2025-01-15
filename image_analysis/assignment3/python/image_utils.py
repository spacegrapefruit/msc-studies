import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Union

from libtiff import TIFF


def load_tiff_image(file_path: str) -> np.ndarray:
    """
    Load a grayscale TIFF image from a file.
    """
    tif = TIFF.open(file_path, mode="r")
    image = tif.read_image()
    tif.close()

    assert image.ndim == 2, "Only grayscale images are supported."
    assert image.dtype == np.uint8, "Only 8-bit images are supported."

    return image


def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Apply a threshold to a grayscale image.
    """
    return image > threshold


def apply_local_threshold(image: np.ndarray, window_size: int, c: int) -> np.ndarray:
    """
    Apply a local threshold to a grayscale image.
    """
    padded_image = np.pad(image, window_size // 2, mode="constant", constant_values=0)

    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        padded_image, (window_size, window_size)
    )

    local_thresholds = np.mean(sliding_windows, axis=(-2, -1)) + c

    return image > local_thresholds


def save_image(image: Union[np.ndarray, plt.Figure], file_path: str):
    """
    Save an image to a file.
    """
    if isinstance(image, plt.Figure):
        image.savefig(file_path)
        return

    if image.dtype == bool:
        image = (image * 255).astype(np.uint8)

    cmap = "gray" if image.ndim == 2 else None
    plt.imsave(file_path, image, cmap=cmap, vmin=0, vmax=255)
    logging.info(f"Saved image to {file_path}")


def flood_fill(
    binary_mask: np.ndarray, labeled_image: np.ndarray, y: int, x: int, label: Any
):
    """
    Flood-fill algorithm to label connected components in a binary mask.
    Modifies the labeled_image in-place.
    """
    stack = [(y, x)]
    while stack:
        cy, cx = stack.pop()
        if binary_mask[cy, cx] and labeled_image[cy, cx] == 0:
            labeled_image[cy, cx] = label
            # 4-connectivity, consider 8-connectivity?
            for ny, nx in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]:
                if 0 <= ny < binary_mask.shape[0] and 0 <= nx < binary_mask.shape[1]:
                    stack.append((ny, nx))


def fill_object_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside binary objects in a binary mask.
    """
    binary_mask = binary_mask.copy()
    padded_mask = np.pad(~binary_mask, 1, mode="constant", constant_values=True)
    background_mask = np.zeros_like(padded_mask, dtype=bool)

    flood_fill(padded_mask, background_mask, 0, 0, label=True)
    background_mask = background_mask[1:-1, 1:-1]
    binary_mask[~background_mask] = True

    return binary_mask


def label_connected_components(binary_mask: np.ndarray) -> np.ndarray:
    """
    Label connected components in a binary mask.
    """
    binary_mask = binary_mask.copy()  # not necessary but makes me sleep better
    labeled_image = np.zeros_like(binary_mask, dtype=np.int32)
    label = 0

    for y, x in np.argwhere(binary_mask):
        if labeled_image[y, x] == 0:
            label += 1
            flood_fill(binary_mask, labeled_image, y, x, label)

    return labeled_image, label


def check_border_touching(binary_mask: np.ndarray, axes: list[int] = [0, 1]) -> bool:
    """
    Check if a binary mask is touching the image border.
    """
    if 0 in axes and np.any(binary_mask[0, :] | binary_mask[-1, :]):
        return True
    if 1 in axes and np.any(binary_mask[:, 0] | binary_mask[:, -1]):
        return True
    return False


def median_filter(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a median filter to an image
    """
    assert window_size % 2 == 1, "Window size must be odd."

    padded_image = np.pad(image, pad_width=window_size // 2, mode="edge")

    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        padded_image, (window_size, window_size)
    )

    filtered_image = np.median(sliding_windows, axis=(-2, -1))

    return filtered_image


def morphological_dilate(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological dilation to a binary mask.
    Uses a square kernel but a very fast implementation.
    """
    rtop, rbottom = (kernel_size + 1) // 2, kernel_size // 2

    padded_mask = np.zeros(
        (binary_mask.shape[0] + kernel_size, binary_mask.shape[1] + kernel_size),
        dtype=binary_mask.dtype,
    )
    padded_mask[rtop:-rbottom, rtop:-rbottom] = binary_mask

    prefix_sum = padded_mask.cumsum(axis=0).cumsum(axis=1)

    total_sum = (
        prefix_sum[kernel_size:, kernel_size:]  # bottom-right corner
        - prefix_sum[kernel_size:, :-kernel_size]  # bottom-left corner
        - prefix_sum[:-kernel_size, kernel_size:]  # top-right corner
        + prefix_sum[:-kernel_size, :-kernel_size]  # top-left corner
    )

    dilated_mask = (total_sum > 0).astype(binary_mask.dtype)
    return dilated_mask


def morphological_erode(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological erosion to a binary mask.
    Uses a square kernel but a very fast implementation.
    """
    rtop, rbottom = (kernel_size + 1) // 2, kernel_size // 2

    padded_mask = np.zeros(
        (binary_mask.shape[0] + kernel_size, binary_mask.shape[1] + kernel_size),
        dtype=binary_mask.dtype,
    )
    padded_mask[rtop:-rbottom, rtop:-rbottom] = binary_mask

    prefix_sum = padded_mask.cumsum(axis=0).cumsum(axis=1)

    total_sum = (
        prefix_sum[kernel_size:, kernel_size:]  # bottom-right corner
        - prefix_sum[kernel_size:, :-kernel_size]  # bottom-left corner
        - prefix_sum[:-kernel_size, kernel_size:]  # top-right corner
        + prefix_sum[:-kernel_size, :-kernel_size]  # top-left corner
    )

    eroded_mask = (total_sum == kernel_size * kernel_size).astype(binary_mask.dtype)
    return eroded_mask


def morphological_open(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological opening to a binary mask.
    """
    return morphological_dilate(
        morphological_erode(binary_mask, kernel_size), kernel_size
    )


def morphological_close(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply morphological closing to a binary mask.
    """
    return morphological_erode(
        morphological_dilate(binary_mask, kernel_size), kernel_size
    )


def calculate_signal_counts(
    labeled_cells: np.ndarray,
    labeled_acridine: np.ndarray,
    labeled_fitc: np.ndarray,
) -> list[dict]:
    """
    Calculate the number of acridine and FITC signals in each cell.
    """
    fitc_counts = np.zeros(np.max(labeled_cells) + 1)
    acridine_counts = np.zeros(np.max(labeled_cells) + 1)

    for i in range(1, np.max(labeled_acridine) + 1):
        which_cell_ids = labeled_cells[labeled_acridine == i].flatten()
        which_cell_ids = np.unique(which_cell_ids[which_cell_ids > 0])

        if len(which_cell_ids) == 1:
            acridine_counts[which_cell_ids[0]] += 1
        elif len(which_cell_ids) > 1:
            logging.warning(
                f"Acridine signal {i} detected in multiple cells: {which_cell_ids}."
            )

    for i in range(1, np.max(labeled_fitc) + 1):
        which_cell_ids = labeled_cells[labeled_fitc == i].flatten()
        which_cell_ids = np.unique(which_cell_ids[which_cell_ids > 0])

        if len(which_cell_ids) == 1:
            fitc_counts[which_cell_ids[0]] += 1
        elif len(which_cell_ids) > 1:
            logging.warning(
                f"FITC signal {i} detected in multiple cells: {which_cell_ids}."
            )

    results = [
        {
            **dict(zip(["y", "x"], calculate_centroid(labeled_cells == i))),
            "cell_area": np.sum(labeled_cells == i),
            "acridine_count": acridine_counts[i],
            "fitc_count": fitc_counts[i],
            "acridine_to_fitc_ratio": acridine_counts[i] / fitc_counts[i]
            if fitc_counts[i] > 0
            else None,
        }
        for i in range(1, np.max(labeled_cells) + 1)
        if labeled_cells[labeled_cells == i].size > 0
    ]

    return results


def make_histogram(
    image: np.ndarray,
    bins: int = 256,
    range: tuple = (0, 256),
    vline=None,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Make a histogram plot of a grayscale image.
    """
    histogram, _ = np.histogram(image, bins=bins, range=range)

    fig, ax = plt.subplots()
    ax.plot(histogram)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Image Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    if vline is not None:
        ax.axvline(vline, color="red")

    return fig


def make_mean_intensity_plot(
    row_means: np.ndarray,
    means_above: np.ndarray,
    means_below: np.ndarray,
    liquid_level: int,
) -> plt.Figure:
    """
    Make a plot of the mean intensity of rows in an image.
    """
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(row_means)
    ax[0].set_ylabel("Row mean intensity")
    ax[1].plot(means_above)
    ax[1].plot(means_below)
    ax[1].plot(means_above - means_below)
    ax[1].set_ylabel("Means")

    return fig


def make_bottle_levels_plot(
    bottle_width: np.ndarray,
    angle1: np.ndarray,
    angle2: np.ndarray,
    liquid_level: int,
    shoulder_level: int,
    neck_level: int,
) -> plt.Figure:
    """
    Make a plot of the bottle width and angles at different levels.
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    # plot bottle width
    ax[0].plot(bottle_width)
    ax[0].grid()

    # plot angles
    ax[1].plot(angle1)
    ax[1].plot(angle2)

    # add lines at all levels
    ax[0].axvline(liquid_level, color="blue")
    ax[0].axvline(shoulder_level, color="red")
    ax[0].axvline(neck_level, color="green")
    ax[1].axvline(liquid_level, color="blue")
    ax[1].axvline(shoulder_level, color="red")
    ax[1].axvline(neck_level, color="green")
    ax[1].grid()

    return fig


def pad_and_clip_angles(angles: np.ndarray, step_size: int = 5) -> np.ndarray:
    """
    Pad and clip angles to avoid edge effects.
    """
    angles = np.pad(angles, step_size, mode="constant", constant_values=0)
    angles = np.maximum(angles, 0)
    return angles


def calculate_gradients(
    bottle_width: np.ndarray, step_size: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gradients of the bottle width.
    """
    angle_top = np.arctan(
        (bottle_width[step_size:-step_size] - bottle_width[: -step_size * 2])
        / step_size
    )
    angle_bottom = np.arctan(
        (bottle_width[step_size * 2 :] - bottle_width[step_size:-step_size]) / step_size
    )
    return angle_top, angle_bottom


def calculate_centroid(region_mask: np.ndarray) -> tuple:
    """
    Calculate the centroid of a region mask.
    """
    coords = np.argwhere(region_mask)
    centroid = np.mean(coords, axis=0)
    return tuple(centroid)


def calculate_dimensions(binary_mask: np.ndarray) -> tuple:
    """
    Calculate the dimensions of a binary mask.
    """
    dim1 = (binary_mask.sum(axis=1) > 0).sum()
    dim2 = (binary_mask.sum(axis=0) > 0).sum()
    return (dim1, dim2)


def fix_salt_and_pepper_noise(
    image: np.ndarray, noise_values: list[int] = [0, 255], window_size: int = 3
) -> np.ndarray:
    """
    Fix salt-and-pepper noise in an image.
    """
    image = image.copy()
    noise_mask = np.isin(image, noise_values)
    image_filtered = median_filter(image, window_size=window_size)
    image[noise_mask] = image_filtered[noise_mask]
    return image


def determine_region_shape(binary_mask, min_dim, max_dim):
    """
    Determine the shape of a soldering region mask (round or square).
    """
    area = binary_mask.sum()
    fill_frac = area / (min_dim * max_dim)

    return "round" if fill_frac < 0.9 else "square"


def make_expected_region_mask(
    mask_shape: tuple[int, int],
    shape_name: str,
    centroid: tuple[int, int],
    *,
    radius: int = None,
    height: int = None,
    width: int = None,
) -> np.ndarray:
    """
    Make an expected region mask for a soldering region.
    """
    expected_mask = np.zeros(mask_shape, dtype=bool)

    if shape_name == "round":
        assert radius is not None
        expected_mask[
            np.linalg.norm(
                np.indices(expected_mask.shape) - np.expand_dims(centroid, axis=[1, 2]),
                axis=0,
            )
            < radius
        ] = 1
    elif shape_name == "square":
        assert height is not None and width is not None
        y_mask = np.abs(np.arange(expected_mask.shape[0]) - centroid[0]) < height / 2
        x_mask = np.abs(np.arange(expected_mask.shape[1]) - centroid[1]) < width / 2
        expected_mask[np.ix_(y_mask, x_mask)] = 1
    else:
        raise ValueError("Invalid shape.")

    return expected_mask
