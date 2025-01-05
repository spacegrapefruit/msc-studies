import logging

import numpy as np
from image_utils import (
    apply_threshold,
    calculate_gradients,
    calculate_signal_counts,
    check_border_touching,
    fill_object_holes,
    label_connected_components,
    load_tiff_image,
    make_bottle_levels_plot,
    make_histogram,
    make_mean_intensity_plot,
    median_filter,
    morphological_dilate,
    morphological_erode,
    pad_and_clip_angles,
    save_image,
    # calculate_centroid,
    # is_centered,
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
        is_touching_border = check_border_touching(cell_mask, axes=[0, 1])
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


def circuit_board_qa_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)

    # fix salt-and-pepper noise using median filtering
    noise_mask = (image == 0) | (image == 255)
    image_filtered = median_filter(image, kernel_size=1)
    image[noise_mask] = image_filtered[noise_mask]

    save_image(image, output_dir / "median_filtered_image.png")

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

#     logging.info(f"Number of detected regions: {num_labels}")

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

#     logging.info("\nDetected soldering regions:")
#     for idx, area, centroid in soldering_regions:
#         logging.info(f"  Region {idx}: Area = {area}, Centroid = {centroid}")

#     logging.info("\nDetected drilled holes:")
#     for idx, area, centroid in holes:
#         logging.info(f"  Hole {idx}: Area = {area}, Centroid = {centroid}")

#     # Check for centering of holes in soldering regions
#     for hole_idx, _, hole_centroid in holes:
#         centered = False
#         for solder_idx, _, solder_centroid in soldering_regions:
#             if is_centered(hole_centroid, solder_centroid):
#                 centered = True
#                 break

#         if not centered:
#             logging.info(f"Warning: Hole {hole_idx} is not centered in any soldering region.")

#     # Check for broken wires (simple adjacency check)
#     # This part is implementation-specific and will depend on the image details


def filled_bottles_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)
    final_image = np.zeros_like(image, dtype=np.uint8)

    histogram_plot = make_histogram(image, log_scale=True)
    save_image(histogram_plot, output_dir / "histogram.png")

    bottle_mask = apply_threshold(image, threshold=0.1)  # TODO: autodetect threshold

    labeled_bottles, num_bottles = label_connected_components(bottle_mask)

    results = []
    for bottle_idx in range(1, num_bottles + 1):
        bottle_mask = labeled_bottles == bottle_idx

        # filter out incomplete cells on the image border
        is_touching_border = check_border_touching(bottle_mask, axes=[1])
        if is_touching_border:
            final_image[bottle_mask] = 64
            continue

        # erode the edges a little to remove black borders
        bottle_mask = morphological_erode(bottle_mask, kernel_size=2)

        # calculate row means
        row_means = np.ma.array(image, mask=~bottle_mask).mean(axis=1)
        # TODO consider using 50-100 edge pixels
        means_above = np.array([row_means[:i].mean() for i in range(1, len(row_means))])
        means_below = np.array([row_means[i:].mean() for i in range(1, len(row_means))])
        means_diff = np.nan_to_num(means_above - means_below, nan=0)
        liquid_level = np.argmax(means_diff)

        mean_intensity_plot = make_mean_intensity_plot(
            row_means, means_above, means_below, liquid_level
        )
        save_image(
            mean_intensity_plot, output_dir / f"bottle_{bottle_idx}_mean_intensity.png"
        )

        bottle_width = bottle_mask.sum(axis=1)
        bottle_width = np.convolve(bottle_width, np.ones(5) / 5, mode="full")
        bottle_width_frac = bottle_width / bottle_width.max()

        angle_top, angle_bottom = calculate_gradients(bottle_width, step_size=5)
        angle1 = pad_and_clip_angles(angle_top - np.abs(angle_bottom), step_size=5)
        angle2 = pad_and_clip_angles(angle_bottom - np.abs(angle_top), step_size=5)

        shoulder_level = np.argmax(angle1 * (0.9 < bottle_width_frac))
        neck_level = np.argmax(
            angle2 * (0.4 < bottle_width_frac) * (bottle_width_frac < 0.6)
        )

        bottle_levels_plot = make_bottle_levels_plot(
            bottle_width=bottle_width,
            angle1=angle1,
            angle2=angle2,
            liquid_level=liquid_level,
            shoulder_level=shoulder_level,
            neck_level=neck_level,
        )
        save_image(
            bottle_levels_plot, output_dir / f"bottle_{bottle_idx}_bottle_levels.png"
        )

        # plot the liquid level
        final_image[bottle_mask] = 255
        liquid_mask = np.zeros_like(bottle_mask)
        liquid_mask[liquid_level:] = 1
        final_image[bottle_mask & liquid_mask] = 128

        # draw lines at shoulder and neck levels
        image_x, image_y = np.indices(bottle_mask.shape)
        final_image[(image_x == shoulder_level) & bottle_mask] = 64
        final_image[(image_x == neck_level) & bottle_mask] = 64

        results.append(
            {
                "bottle_id": bottle_idx,
                "liquid_level": liquid_level,
                "shoulder_level": shoulder_level,
                "neck_level": neck_level,
                "is_filled": liquid_level < (shoulder_level + neck_level) / 2,
            }
        )

    # save final image
    save_image(final_image, output_dir / "final_image.png")

    return results
