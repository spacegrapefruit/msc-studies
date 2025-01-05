import logging

import numpy as np
from image_utils import (
    apply_threshold,
    calculate_signal_counts,
    check_border_touching,
    fill_object_holes,
    label_connected_components,
    load_tiff_image,
    morphological_dilate,
    morphological_erode,
    save_image,
    #     flood_fill,
    #     check_filled_bottles,
    #     visualize_results,
    #     calculate_centroid,
    #     is_centered,
    #     median_filter,
)


MIN_CELL_AREA = 100


def fish_signal_counts_pipeline(
    input_path_acridine: str,
    input_path_fitc: str,
    input_path_dapi: str,
    output_dir: str,
    threshold_acridine: int,
    threshold_fitc: int,
    threshold_dapi: int,
):
    # load images
    acridine_image = load_tiff_image(input_path_acridine)
    fitc_image = load_tiff_image(input_path_fitc)
    dapi_image = load_tiff_image(input_path_dapi)

    # apply thresholds
    dapi_mask = apply_threshold(dapi_image, threshold_dapi)
    acridine_mask = apply_threshold(acridine_image, threshold_acridine)
    fitc_mask = apply_threshold(fitc_image, threshold_fitc)

    # save binary masks
    image = np.stack([dapi_mask, acridine_mask, fitc_mask], axis=-1)
    save_image(image, output_dir / "binary_masks.png")

    # fill holes in DAPI cells
    dapi_mask = fill_object_holes(dapi_mask)

    # label all connected components
    labeled_cells, num_cells = label_connected_components(dapi_mask)
    labeled_acridine, _ = label_connected_components(acridine_mask)
    labeled_fitc, _ = label_connected_components(fitc_mask)

    for cell_idx in range(1, num_cells + 1):
        cell_mask = labeled_cells == cell_idx

        # temporarily remove cell from labeled image
        labeled_cells[cell_mask] = 0
        dapi_mask[cell_mask] = 0

        # morphological erosion of cell mask
        cell_mask = morphological_erode(cell_mask, kernel_size=2)
        # morphological dilation of cell mask
        cell_mask = morphological_dilate(cell_mask, kernel_size=2)

        if not np.any(cell_mask):
            continue

        # filter out incomplete cells on the image border
        is_touching_border = check_border_touching(cell_mask)
        if is_touching_border:
            # half-intensity mask
            dapi_mask[cell_mask] = 128
        else:
            labeled_cells[cell_mask] = cell_idx
            dapi_mask[cell_mask] = 255

        # check for small cells
        cell_area = np.sum(cell_mask)
        if cell_area < MIN_CELL_AREA:
            logging.warning(f"Cell {cell_idx} has area {cell_area} < {MIN_CELL_AREA}.")

    # save binary masks
    image = np.stack([dapi_mask, acridine_mask, fitc_mask], axis=-1)
    save_image(image, output_dir / "binary_masks_cleaned.png")

    # calculate signal counts and ratios
    results = calculate_signal_counts(labeled_cells, labeled_acridine, labeled_fitc)

    return results


#     # # Optional: Display labeled cells
#     # plt.imshow(labeled_cells, cmap="nipy_spectral")
#     # plt.title("Labeled DAPI Cells")
#     # plt.colorbar()
#     # plt.show()


def circuit_board_qa_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)


#     # # Display the original image for reference
#     # plt.figure(figsize=(8, 8))
#     # plt.title("Original Circuit Board X-ray Image")
#     # plt.imshow(image, cmap='gray')
#     # plt.axis('off')
#     # plt.show()

#     # Fix salt-and-pepper noise using median filtering
#     image = median_filter(image, kernel_size=3)

#     # Thresholding to identify bright regions (drilled holes, soldering regions)
#     threshold = np.percentile(image, 99)  # Bright regions threshold
#     binary_image = image > threshold

#     # # Display binary image
#     # plt.figure(figsize=(8, 8))
#     # plt.title("Binary Image (Bright Regions)")
#     # plt.imshow(binary_image, cmap='gray')
#     # plt.axis('off')
#     # plt.show()

#     # Find connected components (regions)
#     labeled_image, num_labels = label_connected_components(binary_image)

#     print(f"Number of detected regions: {num_labels}")

#     # Analyze regions for size, shape, and centering
#     soldering_regions = []
#     holes = []

#     for region_idx in range(1, num_labels + 1):  # Exclude background (label 0)
#         region_mask = labeled_image == region_idx

#         # Region properties
#         area = np.sum(region_mask)
#         centroid = calculate_centroid(region_mask)

#         # Classify region
#         if area > 100:  # Example size threshold for soldering regions
#             soldering_regions.append((region_idx, area, centroid))
#         elif area <= 100:  # Example size threshold for drilled holes
#             holes.append((region_idx, area, centroid))

#     print("\nDetected soldering regions:")
#     for idx, area, centroid in soldering_regions:
#         print(f"  Region {idx}: Area = {area}, Centroid = {centroid}")

#     print("\nDetected drilled holes:")
#     for idx, area, centroid in holes:
#         print(f"  Hole {idx}: Area = {area}, Centroid = {centroid}")

#     # Check for centering of holes in soldering regions
#     for hole_idx, _, hole_centroid in holes:
#         centered = False
#         for solder_idx, _, solder_centroid in soldering_regions:
#             if is_centered(hole_centroid, solder_centroid):
#                 centered = True
#                 break

#         if not centered:
#             print(f"Warning: Hole {hole_idx} is not centered in any soldering region.")

#     # Check for broken wires (simple adjacency check)
#     # This part is implementation-specific and will depend on the image details


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=sigma)


def filled_bottles_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)

    #     image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # gaussian filter
    # image = gaussian_blur(image, sigma=1)

    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    # save histogram
    import matplotlib.pyplot as plt

    plt.plot(histogram)
    plt.yscale("log")
    plt.title("Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "histogram.png")

    bottle_mask = apply_threshold(image, threshold=0.1)  # TODO: autodetect threshold

    labeled_bottles, num_bottles = label_connected_components(bottle_mask)

    for bottle_idx in range(1, num_bottles + 1):
        bottle_mask = labeled_bottles == bottle_idx


#         centroid = calculate_centroid(region_mask)
#         print(f"  Bottle {i}: Centroid = {centroid}")

#     improperly_filled = check_filled_bottles(preprocessed_image, labeled_image)
#     # visualize_results(preprocessed_image, labeled_bottles, improperly_filled)
#     return improperly_filled
