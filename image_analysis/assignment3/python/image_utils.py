import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

from libtiff import TIFF


def load_tiff_image(file_path: str) -> np.ndarray:
    tif = TIFF.open(file_path, mode="r")
    image = tif.read_image()
    tif.close()

    assert image.ndim == 2, "Only grayscale images are supported."
    assert image.dtype == np.uint8, "Only 8-bit images are supported."

    return image


def apply_threshold(image: object, threshold: int) -> np.ndarray:
    return (image > threshold).astype(np.uint8) * 255


def save_image(image: Union[np.ndarray, plt.Figure], file_path: str):
    if isinstance(image, plt.Figure):
        image.savefig(file_path)
    else:
        cmap = "gray" if image.ndim == 2 else None
        plt.imsave(file_path, image, cmap=cmap, vmin=0, vmax=255)
    logging.info(f"Saved image to {file_path}")


# 4-connectivity, TODO: 8-connectivity?
def flood_fill(binary_mask, labeled_image, x, y, label):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if binary_mask[cx, cy] and labeled_image[cx, cy] == 0:
            labeled_image[cx, cy] = label
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                if 0 <= nx < binary_mask.shape[0] and 0 <= ny < binary_mask.shape[1]:
                    stack.append((nx, ny))


def fill_object_holes(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = binary_mask.copy()
    padded_mask = np.pad(~binary_mask, 1, mode="constant", constant_values=255)
    background_mask = np.zeros_like(padded_mask, dtype=bool)

    flood_fill(padded_mask, background_mask, 0, 0, True)
    background_mask = background_mask[1:-1, 1:-1]
    binary_mask[~background_mask] = 255

    return binary_mask


def label_connected_components(binary_mask: np.ndarray) -> np.ndarray:
    labeled_image = np.zeros_like(binary_mask, dtype=np.int32)
    label = 0
    rows, cols = binary_mask.shape

    for x in range(cols):
        for y in range(rows):
            if binary_mask[y, x] and labeled_image[y, x] == 0:
                label += 1
                flood_fill(binary_mask, labeled_image, y, x, label)

    return labeled_image, label


def check_border_touching(binary_mask: np.ndarray, axes=[0, 1]) -> bool:
    if 0 in axes and np.any(binary_mask[0, :] | binary_mask[-1, :]):
        return True
    if 1 in axes and np.any(binary_mask[:, 0] | binary_mask[:, -1]):
        return True
    return False


def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    window_size = 2 * kernel_size + 1

    padded_image = np.pad(image, pad_width=kernel_size, mode="edge")

    sliding_windows = np.lib.stride_tricks.sliding_window_view(
        padded_image, (window_size, window_size)
    )

    filtered_image = np.median(sliding_windows, axis=(-2, -1))

    return filtered_image


def morphological_dilate(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return ~(median_filter(~binary_mask, kernel_size=2)).astype(bool)


def morphological_erode(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return median_filter(binary_mask, kernel_size=2).astype(bool)


def calculate_signal_counts(
    labeled_cells: np.ndarray,
    labeled_acridine: np.ndarray,
    labeled_fitc: np.ndarray,
):
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
            "cell_id": i,
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
    image: np.ndarray, bins: int = 256, range: tuple = (0, 256), log_scale: bool = False
) -> plt.Figure:
    histogram, _ = np.histogram(image, bins=bins, range=range)

    fig, ax = plt.subplots()
    ax.plot(histogram)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Image Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")

    return fig


def make_mean_intensity_plot(
    row_means: np.ndarray,
    means_above: np.ndarray,
    means_below: np.ndarray,
    liquid_level: int,
) -> plt.Figure:
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
):
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


def pad_and_clip_angles(angles: np.ndarray, step_size: int = 5):
    angles = np.pad(angles, step_size, mode="constant", constant_values=0)
    angles = np.maximum(angles, 0)
    return angles


def calculate_gradients(bottle_width: np.ndarray, step_size: int = 5):
    angle_top = np.arctan(
        (bottle_width[step_size:-step_size] - bottle_width[: -step_size * 2])
        / step_size
    )
    angle_bottom = np.arctan(
        (bottle_width[step_size * 2 :] - bottle_width[step_size:-step_size]) / step_size
    )
    return angle_top, angle_bottom


# def detect_edges(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
#     gradient = np.abs(np.gradient(image, axis=0)) + np.abs(np.gradient(image, axis=1))
#     edges = gradient > threshold
#     return edges


# def calculate_centroid(region_mask):
#     coords = np.argwhere(region_mask)
#     centroid = np.mean(coords, axis=0)
#     return tuple(centroid)


# def is_centered(hole_centroid, solder_centroid, tolerance=5):
#     distance = np.linalg.norm(np.array(hole_centroid) - np.array(solder_centroid))
#     return distance <= tolerance
