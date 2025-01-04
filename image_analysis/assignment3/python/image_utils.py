import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from libtiff import TIFF


def load_tiff_image(file_path: str) -> np.ndarray:
    tiff = TIFF.open(file_path, mode="r")
    image = tiff.read_image()
    tiff.close()
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:  # Convert RGB to grayscale if needed
        image = np.mean(image, axis=2)

    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return normalized_image


def preprocess_image2(image: object, threshold: int) -> np.ndarray:
    image = np.array(image, dtype=np.uint8)
    binary_mask = (image > threshold).astype(np.uint8) * 255
    return binary_mask


def save_image(image: np.ndarray, file_path: str):
    if image.ndim == 1:
        plt.imsave(file_path, image, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imsave(file_path, image, vmin=0, vmax=255)


def calculate_signal_counts(
    labeled_cells: np.ndarray, labeled_acridine: np.ndarray, labeled_fitc: np.ndarray,
):
    fitc_counts = np.zeros(np.max(labeled_cells) + 1)
    acridine_counts = np.zeros(np.max(labeled_cells) + 1)

    for i in range(1, np.max(labeled_acridine) + 1):
        which_cell_ids = labeled_cells[labeled_acridine == i].flatten()
        which_cell_ids = np.unique(which_cell_ids[which_cell_ids > 0])

        if len(which_cell_ids) != 1:
            continue
        acridine_counts[which_cell_ids[0]] += 1

    for i in range(1, np.max(labeled_fitc) + 1):
        which_cell_ids = labeled_cells[labeled_fitc == i].flatten()
        which_cell_ids = np.unique(which_cell_ids[which_cell_ids > 0])

        if len(which_cell_ids) != 1:
            continue
        fitc_counts[which_cell_ids[0]] += 1

    results = [
        {
            "cell_id": i,
            "acridine_count": acridine_counts[i],
            "fitc_count": fitc_counts[i],
            "acridine_to_fitc_ratio": acridine_counts[i] / fitc_counts[i] if fitc_counts[i] > 0 else None,
        }
        for i in range(1, np.max(labeled_cells) + 1)
        if labeled_cells[labeled_cells == i].size > 0
    ]

    return results


def segment_bottles(image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Segment bottles by thresholding light regions."""
    bottle_mask = image > threshold
    return bottle_mask


def detect_edges(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    gradient = np.abs(np.gradient(image, axis=0)) + np.abs(np.gradient(image, axis=1))
    edges = gradient > threshold
    return edges


def check_filled_bottles(image: np.ndarray, labeled_image: np.ndarray) -> list:
    improperly_filled_bottles = []

    for i in range(1, np.max(labeled_image) + 1):
        region_mask = labeled_image == i
        region_dims = np.argwhere(region_mask)
        bottle_corners = (
            np.min(region_dims[:, 0]),
            np.max(region_dims[:, 0]),
            np.min(region_dims[:, 1]),
            np.max(region_dims[:, 1]),
        )
        print(f"Region {i} corners: {bottle_corners}")

        # Check if the bottle is filled
        bottle_region = image[bottle_corners[0] : bottle_corners[1], bottle_corners[2] : bottle_corners[3]]
        colour_levels = np.zeros((bottle_corners[1] - bottle_corners[0], 3))  # white, grey, black
        for row in range(bottle_corners[0], bottle_corners[1]):
            colour_levels[row - bottle_corners[0], 0] = np.sum(bottle_region[row - bottle_corners[0]] > 0.95)
            colour_levels[row - bottle_corners[0], 2] += np.sum(bottle_region[row - bottle_corners[0]] == 0)
            colour_levels[row - bottle_corners[0], 1] = bottle_corners[3] - bottle_corners[2] - colour_levels[row - bottle_corners[0], 0] - colour_levels[row - bottle_corners[0], 2] + 1
        print(f"Colour levels: {colour_levels}")

    quit()
    return improperly_filled_bottles


def visualize_results(image: np.ndarray, bottle_regions: list, improperly_filled: list):
    plt.imshow(image, cmap="gray")

    for region in bottle_regions:
        col, start, end = region
        color = "g" if region not in improperly_filled else "r"
        plt.plot([col, col], [start, end], color)

    plt.title("Bottle Fill Detection")
    plt.show()


# TODO directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
def flood_fill(binary_mask, labeled_image, x, y, label):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if binary_mask[cx, cy] and labeled_image[cx, cy] == 0:
            labeled_image[cx, cy] = label
            for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                if 0 <= nx < binary_mask.shape[0] and 0 <= ny < binary_mask.shape[1]:
                    stack.append((nx, ny))


def label_connected_components(binary_mask: np.ndarray) -> np.ndarray:
    labeled_image = np.zeros_like(binary_mask, dtype=np.int32)
    label = 1
    rows, cols = binary_mask.shape

    for x in range(rows):
        for y in range(cols):
            if binary_mask[x, y] and labeled_image[x, y] == 0:
                flood_fill(binary_mask, labeled_image, x, y, label)
                label += 1

    return labeled_image, label - 1


def measure_connected_components(binary_image):
    labeled_image, num_labels = label_connected_components(binary_image)
    return labeled_image, num_labels


def calculate_centroid(region_mask):
    coords = np.argwhere(region_mask)
    centroid = np.mean(coords, axis=0)
    return tuple(centroid)


def is_centered(hole_centroid, solder_centroid, tolerance=5):
    distance = np.linalg.norm(np.array(hole_centroid) - np.array(solder_centroid))
    return distance <= tolerance


def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    # Compute the total kernel size
    window_size = 2 * kernel_size + 1

    # Pad the image to handle edges
    padded_image = np.pad(image, pad_width=kernel_size, mode='edge')

    # Create sliding window views
    sliding_windows = np.lib.stride_tricks.sliding_window_view(padded_image, (window_size, window_size))

    # Compute the median across the last two dimensions (kernel dimensions)
    filtered_image = np.median(sliding_windows, axis=(-2, -1))

    return filtered_image
