import numpy as np
from libtiff import TIFF
from image_utils import (
    load_tiff_image,
    flood_fill,
    preprocess_image,
    preprocess_image2,
    segment_bottles,
    check_filled_bottles,
    visualize_results,
    measure_connected_components,
    calculate_centroid,
    is_centered,
    label_connected_components,
    calculate_signal_counts,
    median_filter,
    save_image,
)


def fish_signal_counts_pipeline(
    input_path_acridine: str,
    input_path_fitc: str,
    input_path_dapi: str,
    output_dir: str,
    *,
    threshold_acridine: int = 128,
    threshold_fitc: int = 128,
    threshold_dapi: int = 30,
):
    # load images
    acridine_image = load_tiff_image(input_path_acridine)
    fitc_image = load_tiff_image(input_path_fitc)
    dapi_image = load_tiff_image(input_path_dapi)

    # preprocess images to create binary masks
    dapi_mask = preprocess_image2(dapi_image, threshold_dapi)
    acridine_mask = preprocess_image2(acridine_image, threshold_acridine)
    fitc_mask = preprocess_image2(fitc_image, threshold_fitc)

    # save binary masks
    image = np.stack([dapi_mask, acridine_mask, fitc_mask], axis=-1)
    save_image(image, output_dir / "binary_masks.png")

    # fill holes in DAPI cells
    dapi_mask_padded = np.pad(~dapi_mask, 1, mode="constant", constant_values=255)
    dapi_background_mask = np.zeros_like(dapi_mask_padded, dtype=bool)
    flood_fill(dapi_mask_padded, dapi_background_mask, 0, 0, True)
    dapi_background_mask = dapi_background_mask[1:-1, 1:-1]
    dapi_mask[~dapi_background_mask] = 255

    # Label DAPI cells
    labeled_cells, num_cells = label_connected_components(dapi_mask)

    # # filter out small cells (noise)
    # min_cell_area = 100
    for cell_idx in range(1, num_cells + 1):
        cell_mask = labeled_cells == cell_idx
        labeled_cells[cell_mask] = 0  # temporary remove cell from labeled image

        # morphological erosion of cell mask
        new_cell_mask = median_filter(cell_mask, kernel_size=2).astype(bool)
        # morphological dilation of cell mask
        new_cell_mask = ~(median_filter(~new_cell_mask, kernel_size=2)).astype(bool)

        dapi_mask[cell_mask] = 0
        dapi_mask[new_cell_mask] = 255

        if not np.any(new_cell_mask):
            continue

        # filter out incomplete cells
        cell_xy = np.argwhere(new_cell_mask)
        min_x, min_y = np.min(cell_xy, axis=0)
        max_x, max_y = np.max(cell_xy, axis=0)

        if min_x == 0 or min_y == 0 or max_x == new_cell_mask.shape[0] - 1 or max_y == new_cell_mask.shape[1] - 1:
            labeled_cells[new_cell_mask] = 0
            dapi_mask[new_cell_mask] = 128
        else:
            labeled_cells[new_cell_mask] = cell_idx

        # # filter out small cells
        # cell_area = np.sum(cell_mask)
        # if cell_area < min_cell_area:
        #     labeled_cells[cell_mask] = 0
        #     dapi_mask[cell_mask] = 128

    # save binary masks
    image = np.stack([dapi_mask, acridine_mask, fitc_mask], axis=-1)
    save_image(image, output_dir / "binary_masks_cleaned.png")

    num_cells = len(np.unique(labeled_cells)) - 1

    # Label acridine and FITC signals
    labeled_acridine, _ = label_connected_components(acridine_mask)
    labeled_fitc, _ = label_connected_components(fitc_mask)

    # Calculate signal counts and ratios
    results = calculate_signal_counts(labeled_cells, labeled_acridine, labeled_fitc)

    # Display results
    total_acridine = sum(result["acridine_count"] for result in results)
    total_fitc = sum(result["fitc_count"] for result in results)

    print("Cell Analysis Results")
    print("Total Cells:", num_cells)
    print("---------------------")
    for result in results:
        print(
            f"Cell {result['cell_id']}, Area={np.sum(labeled_cells == result['cell_id'])}: Acridine={result['acridine_count']}, FITC={result['fitc_count']}, Ratio={result['acridine_to_fitc_ratio']}"
        )
    print(f"Total Acridine Signals: {total_acridine}")
    print(f"Total FITC Signals: {total_fitc}")

    # # Optional: Display labeled cells
    # plt.imshow(labeled_cells, cmap="nipy_spectral")
    # plt.title("Labeled DAPI Cells")
    # plt.colorbar()
    # plt.show()


def circuit_board_qa_pipeline(input_path: str, output_dir: str):
    # Load the grayscale TIFF image
    tiff = TIFF.open(input_path, mode="r")
    image = tiff.read_image()

    if image.ndim != 2:
        raise ValueError("Input image must be a grayscale image.")

    # # Display the original image for reference
    # plt.figure(figsize=(8, 8))
    # plt.title("Original Circuit Board X-ray Image")
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Fix salt-and-pepper noise using median filtering
    image = median_filter(image, kernel_size=3)

    # Thresholding to identify bright regions (drilled holes, soldering regions)
    threshold = np.percentile(image, 99)  # Bright regions threshold
    binary_image = image > threshold

    # # Display binary image
    # plt.figure(figsize=(8, 8))
    # plt.title("Binary Image (Bright Regions)")
    # plt.imshow(binary_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Find connected components (regions)
    labeled_image, num_labels = measure_connected_components(binary_image)

    print(f"Number of detected regions: {num_labels}")

    # Analyze regions for size, shape, and centering
    soldering_regions = []
    holes = []

    for region_idx in range(1, num_labels + 1):  # Exclude background (label 0)
        region_mask = labeled_image == region_idx

        # Region properties
        area = np.sum(region_mask)
        centroid = calculate_centroid(region_mask)

        # Classify region
        if area > 100:  # Example size threshold for soldering regions
            soldering_regions.append((region_idx, area, centroid))
        elif area <= 100:  # Example size threshold for drilled holes
            holes.append((region_idx, area, centroid))

    print("\nDetected soldering regions:")
    for idx, area, centroid in soldering_regions:
        print(f"  Region {idx}: Area = {area}, Centroid = {centroid}")

    print("\nDetected drilled holes:")
    for idx, area, centroid in holes:
        print(f"  Hole {idx}: Area = {area}, Centroid = {centroid}")

    # Check for centering of holes in soldering regions
    for hole_idx, _, hole_centroid in holes:
        centered = False
        for solder_idx, _, solder_centroid in soldering_regions:
            if is_centered(hole_centroid, solder_centroid):
                centered = True
                break

        if not centered:
            print(f"Warning: Hole {hole_idx} is not centered in any soldering region.")

    # Check for broken wires (simple adjacency check)
    # This part is implementation-specific and will depend on the image details


def filled_bottles_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)
    preprocessed_image = preprocess_image(image)
    bottle_mask = segment_bottles(preprocessed_image)

    labeled_image, num_labels = label_connected_components(bottle_mask)

    print(f"Found {num_labels} bottle regions.")
    for i in range(1, num_labels + 1):
        region_mask = labeled_image == i
        centroid = calculate_centroid(region_mask)
        print(f"  Bottle {i}: Centroid = {centroid}")

    improperly_filled = check_filled_bottles(preprocessed_image, labeled_image)
    # visualize_results(preprocessed_image, labeled_bottles, improperly_filled)
    return improperly_filled
