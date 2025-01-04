import numpy as np
from libtiff import TIFF
from image_utils import (
    load_tiff_image,
    preprocess_image,
    preprocess_image2,
    detect_edges,
    find_bottle_regions,
    check_filled_bottles,
    visualize_results,
    measure_connected_components,
    calculate_centroid,
    is_centered,
    label_connected_components,
    calculate_signal_counts,
)


def fish_signal_counts_pipeline(
    input_path_acridine: str,
    input_path_fitc: str,
    input_path_dapi: str,
    threshold_acridine: int = 50,
    threshold_fitc: int = 50,
    threshold_dapi: int = 50,
):
    """Main function to process FISH images and calculate signal counts."""
    # Load images
    acridine_image = load_tiff_image(input_path_acridine)
    fitc_image = load_tiff_image(input_path_fitc)
    dapi_image = load_tiff_image(input_path_dapi)

    # Preprocess images to create binary masks
    dapi_mask = preprocess_image2(dapi_image, threshold_dapi)
    acridine_mask = preprocess_image2(acridine_image, threshold_acridine)
    fitc_mask = preprocess_image2(fitc_image, threshold_fitc)

    # Label DAPI cells
    labeled_cells, num_cells = label_connected_components(dapi_mask)

    # Calculate signal counts and ratios
    results = calculate_signal_counts(labeled_cells, acridine_mask, fitc_mask)

    # Display results
    print("Cell Analysis Results:")
    for result in results:
        print(result)

    # # Optional: Display labeled cells
    # plt.imshow(labeled_cells, cmap="nipy_spectral")
    # plt.title("Labeled DAPI Cells")
    # plt.colorbar()
    # plt.show()


def circuit_board_qa_pipeline(file_path: str):
    # Load the grayscale TIFF image
    tiff = TIFF.open(file_path, mode="r")
    image = tiff.read_image()

    if image.ndim != 2:
        raise ValueError("Input image must be a grayscale image.")

    # # Display the original image for reference
    # plt.figure(figsize=(8, 8))
    # plt.title("Original Circuit Board X-ray Image")
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()

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


def filled_bottles_pipeline(file_path: str):
    image = load_tiff_image(file_path)
    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image)
    bottle_regions = find_bottle_regions(edges)
    improperly_filled = check_filled_bottles(preprocessed_image, bottle_regions)
    # visualize_results(preprocessed_image, bottle_regions, improperly_filled)
    return improperly_filled
