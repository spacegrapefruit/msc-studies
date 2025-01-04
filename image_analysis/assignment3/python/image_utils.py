import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from libtiff import TIFF


def calculate_signal_counts(labeled_cells: np.ndarray, acridine_mask: np.ndarray, fitc_mask: np.ndarray):
    """Calculate the number of Acridine and FITC signals per cell."""
    unique_labels = np.unique(labeled_cells)[1:]  # Exclude background label (0)
    results = []

    for label in unique_labels:
        cell_mask = (labeled_cells == label)
        acridine_count = np.sum(acridine_mask[cell_mask])
        fitc_count = np.sum(fitc_mask[cell_mask])
        ratio = acridine_count / fitc_count if fitc_count > 0 else None
        results.append({
            "cell_id": label,
            "acridine_count": acridine_count,
            "fitc_count": fitc_count,
            "acridine_to_fitc_ratio": ratio,
        })

    return results


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
    binary_mask = (image > threshold).astype(np.uint8)
    return binary_mask


def detect_edges(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    gradient = np.abs(np.gradient(image, axis=0)) + np.abs(np.gradient(image, axis=1))
    edges = gradient > threshold
    return edges


def find_bottle_regions(edges: np.ndarray) -> list:
    bottle_regions = []
    rows, cols = edges.shape

    for col in range(cols):
        column_data = edges[:, col]
        bottle_start = None
        bottle_end = None

        for row in range(rows):
            if column_data[row] and bottle_start is None:
                bottle_start = row
            elif not column_data[row] and bottle_start is not None:
                bottle_end = row
                break

        if bottle_start is not None and bottle_end is not None:
            bottle_regions.append((col, bottle_start, bottle_end))

    return bottle_regions


def check_filled_bottles(image: np.ndarray, bottle_regions: list) -> list:
    improperly_filled_bottles = []

    for region in bottle_regions:
        col, start, end = region
        bottle_height = end - start

        neck_midpoint = start + int(bottle_height / 4)  # Assuming neck is top 25%
        shoulder_midpoint = start + int(bottle_height / 2)
        liquid_level = np.argmax(image[start:end, col] > 0.5) + start

        if liquid_level < (neck_midpoint + shoulder_midpoint) // 2:
            improperly_filled_bottles.append(region)

    return improperly_filled_bottles


def visualize_results(image: np.ndarray, bottle_regions: list, improperly_filled: list):
    plt.imshow(image, cmap="gray")

    for region in bottle_regions:
        col, start, end = region
        color = "g" if region not in improperly_filled else "r"
        plt.plot([col, col], [start, end], color)

    plt.title("Bottle Fill Detection")
    plt.show()


# def label_connected_components(binary_mask: np.ndarray) -> np.ndarray:
#     labeled_image = np.zeros_like(binary_mask, dtype=np.int32)
#     label = 1

#     def flood_fill(x, y):
#         stack = [(x, y)]
#         while stack:
#             cx, cy = stack.pop()
#             if binary_mask[cx, cy] and labeled_image[cx, cy] == 0:
#                 labeled_image[cx, cy] = label
#                 for nx, ny in [
#                     (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)
#                 ]:
#                     if 0 <= nx < binary_mask.shape[0] and 0 <= ny < binary_mask.shape[1]:
#                         stack.append((nx, ny))

#     for x in range(binary_mask.shape[0]):
#         for y in range(binary_mask.shape[1]):
#             if binary_mask[x, y] and labeled_image[x, y] == 0:
#                 flood_fill(x, y)
#                 label += 1

#     return labeled_image


def label_connected_components(binary_image):
    labeled_image = np.zeros_like(binary_image, dtype=int)
    label = 0
    rows, cols = binary_image.shape

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

    for r in range(rows):
        for c in range(cols):
            if binary_image[r, c] and labeled_image[r, c] == 0:
                label += 1
                queue = deque([(r, c)])

                while queue:
                    x, y = queue.popleft()

                    if labeled_image[x, y] == 0:
                        labeled_image[x, y] = label

                        for dr, dc in directions:
                            nx, ny = x + dr, y + dc
                            if 0 <= nx < rows and 0 <= ny < cols and binary_image[nx, ny] and labeled_image[nx, ny] == 0:
                                queue.append((nx, ny))

    return labeled_image, label


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
