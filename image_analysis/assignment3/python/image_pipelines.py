import logging
import warnings

import numpy as np
from image_utils import (
    apply_threshold,
    calculate_centroid,
    calculate_dimensions,
    calculate_gradients,
    calculate_signal_counts,
    check_border_touching,
    determine_region_shape,
    fill_object_holes,
    fix_salt_and_pepper_noise,
    label_connected_components,
    load_tiff_image,
    make_bottle_levels_plot,
    make_expected_region_mask,
    make_histogram,
    make_mean_intensity_plot,
    morphological_erode,
    morphological_open,
    pad_and_clip_angles,
    save_image,
)


class PCBColour:
    FUSE_WIRE = 16
    SOLDER = 64
    SOLDERED_CONNECTION = 80
    SILICON = 128
    MICROCONTROLLER = 176
    HOLES = 240
    # unused: 80, 96


EXPECTED_REGION_HOLE_DIMS = {
    "round": [(7, 7)],
    "square": [(7, 10), (10, 7)],
}


MIN_CELL_AREA = 100


def fish_signal_counts_pipeline(
    input_path_acridine: str,
    input_path_fitc: str,
    input_path_dapi: str,
    output_dir: str,
    threshold_acridine: int,
    threshold_fitc: int,
    threshold_dapi: int,
) -> list[dict]:
    """
    Detects cells in a fish image and calculates signal counts and ratios.
    """
    # load images
    acridine_image = load_tiff_image(input_path_acridine)
    fitc_image = load_tiff_image(input_path_fitc)
    dapi_image = load_tiff_image(input_path_dapi)

    # # save raw images
    # for name, image, threshold in zip(
    #     ["acridine", "fitc", "dapi"],
    #     [acridine_image, fitc_image, dapi_image],
    #     [threshold_acridine, threshold_fitc, threshold_dapi],
    # ):
    #     save_image(image, output_dir / f"{name}_image_raw.png")
    #     histogram_plot = make_histogram(image, vline=threshold, log_scale=True)
    #     save_image(histogram_plot, output_dir / f"{name}_histogram.png")
    save_image(
        np.stack([dapi_image, acridine_image, fitc_image], axis=-1),
        output_dir / "raw_rgb_image.png",
    )

    # apply thresholds
    dapi_mask = apply_threshold(dapi_image, threshold_dapi)
    acridine_mask = apply_threshold(acridine_image, threshold_acridine)
    fitc_mask = apply_threshold(fitc_image, threshold_fitc)

    # save binary masks
    image = np.stack([dapi_mask, acridine_mask, fitc_mask], axis=-1)
    save_image(image, output_dir / "binary_masks.png")

    # fill holes in DAPI cells
    dapi_mask = fill_object_holes(dapi_mask)

    # morphological opening to remove noise
    dapi_mask = morphological_open(dapi_mask, kernel_size=5)

    # # morphological closing to connect fragmented signals
    # acridine_mask = morphological_close(acridine_mask, kernel_size=3)
    # fitc_mask = morphological_close(fitc_mask, kernel_size=3)

    # label all connected components
    labeled_cells, num_cells = label_connected_components(dapi_mask)
    labeled_acridine, _ = label_connected_components(acridine_mask)
    labeled_fitc, _ = label_connected_components(fitc_mask)

    dapi_image = np.zeros_like(dapi_mask, dtype=np.uint8)
    for cell_idx in range(1, num_cells + 1):
        cell_mask = labeled_cells == cell_idx

        # filter out incomplete cells on the image border
        is_touching_border = check_border_touching(cell_mask, axes=[0, 1])
        if is_touching_border:
            # half-intensity mask
            labeled_cells[cell_mask] = 0
            dapi_image[cell_mask] = 128
        else:
            dapi_image[cell_mask] = 255

        # check for small cells
        cell_area = np.sum(cell_mask)
        if cell_area < MIN_CELL_AREA:
            logging.warning(f"Cell {cell_idx} has area {cell_area} < {MIN_CELL_AREA}.")

    # save binary masks
    image = np.stack(
        [dapi_image, acridine_mask * 255, fitc_mask * 255],
        axis=-1,
    ).astype(np.uint8)
    save_image(image, output_dir / "binary_masks_cleaned.png")

    # calculate signal counts and ratios
    results = calculate_signal_counts(labeled_cells, labeled_acridine, labeled_fitc)

    return results


def circuit_board_qa_pipeline(input_path: str, output_dir: str) -> list[dict]:
    """
    Detects defects in a circuit board image.
    """
    image = load_tiff_image(input_path)

    # fix salt-and-pepper noise using median filtering
    image = fix_salt_and_pepper_noise(image, window_size=3)
    save_image(image, output_dir / "image_denoised.png")

    image[image == PCBColour.SOLDERED_CONNECTION] = PCBColour.SOLDER

    holes_mask = image == PCBColour.HOLES
    solder_mask = (image == PCBColour.SOLDER) | holes_mask

    # morphological opening to remove thin traces
    solder_mask_thick = morphological_open(solder_mask, kernel_size=5)
    solder_mask_thick = solder_mask_thick & solder_mask
    save_image(solder_mask_thick, output_dir / "solder_mask_thick.png")

    labeled_components, num_components = label_connected_components(solder_mask)
    labeled_holes, num_holes = label_connected_components(holes_mask)
    holes_checked = np.zeros(num_holes + 1, dtype=bool)

    # processed image for visualisation
    final_image = np.full((*image.shape, 3), PCBColour.SILICON, dtype=np.uint8)
    final_image[solder_mask] = PCBColour.SOLDER + 16  # make traces slightly lighter
    final_image[solder_mask_thick] = PCBColour.SOLDER
    final_image[holes_mask] = PCBColour.HOLES

    results = []
    for component_idx in range(1, num_components + 1):
        component_mask = labeled_components == component_idx
        component_mask_thick = component_mask & solder_mask_thick
        component_centroid = calculate_centroid(component_mask)

        labeled_regions, num_regions = label_connected_components(component_mask_thick)

        has_traces = (component_mask & (~component_mask_thick)).sum() > 0
        if has_traces and num_regions < 2:
            # set green, blue channels to 0
            final_image[component_mask, 1:3] = 0
            results.append(
                {
                    **dict(zip(["y", "x"], component_centroid)),
                    "message": f"Suspected broken wire touches {num_regions} != 2 connectors",
                }
            )

        # check soldering regions
        for region_idx in range(1, num_regions + 1):
            region_mask = labeled_regions == region_idx
            min_dim, max_dim = sorted(calculate_dimensions(region_mask))
            region_centroid = calculate_centroid(region_mask)

            if min_dim < 9 and max_dim < 15:
                # component or contact connector
                continue

            # otherwise we're dealing with a soldering region
            region_shape_name = determine_region_shape(region_mask, min_dim, max_dim)

            expected_mask = make_expected_region_mask(
                mask_shape=region_mask.shape,
                shape_name=region_shape_name,
                centroid=region_centroid,
                radius=(241 / np.pi) ** 0.5,
                height=17,
                width=20,
            )

            intersection = np.sum(region_mask & expected_mask)
            area = region_mask.sum()

            if intersection < max(area, expected_mask.sum()) * 0.95:
                final_image[region_mask, 1:3] = 0
                results.append(
                    {
                        **dict(zip(["y", "x"], region_centroid)),
                        "message": f"{region_shape_name} soldering region, area: {area}, expected: {expected_mask.sum()}, intersection: {intersection}",
                    }
                )

            holes_in_region = sorted(set(np.unique(labeled_holes[region_mask])) - {0})
            holes_checked[holes_in_region] = True
            if len(holes_in_region) != 1:
                results.append(
                    {
                        **dict(zip(["y", "x"], region_centroid)),
                        "message": f"{region_shape_name} soldering region has != 1 holes: {holes_in_region}",
                    }
                )
                continue

            hole_mask = labeled_holes == holes_in_region[0]
            hole_centroid = calculate_centroid(hole_mask)
            hole_dims = calculate_dimensions(hole_mask)
            expected_hole_dims = EXPECTED_REGION_HOLE_DIMS[region_shape_name]
            expected_hole_mask = make_expected_region_mask(
                mask_shape=hole_mask.shape,
                shape_name=region_shape_name,
                centroid=hole_centroid,
                radius=(45 / np.pi) ** 0.5,
                height=7,
                width=10,
            )

            if hole_centroid != region_centroid:
                final_image[hole_mask, 1:3] = 0
                results.append(
                    {
                        **dict(zip(["y", "x"], region_centroid)),
                        "message": f"{region_shape_name} soldering region has hole at centroid: {hole_centroid}, expected: {region_centroid}",
                    }
                )
            if hole_dims not in expected_hole_dims:
                final_image[hole_mask, 1:3] = 0
                results.append(
                    {
                        **dict(zip(["y", "x"], hole_centroid)),
                        "message": f"{region_shape_name} soldering region has hole with dimensions: {hole_dims}, expected one of: {expected_hole_dims}",
                    }
                )

            intersection = np.sum(hole_mask & expected_hole_mask)
            area = hole_mask.sum()

            if intersection < max(area, expected_hole_mask.sum()) * 0.95:
                final_image[hole_mask, 1:3] = 0
                results.append(
                    {
                        **dict(zip(["y", "x"], hole_centroid)),
                        "message": f"{region_shape_name} hole, area: {area}, expected: {expected_hole_mask.sum()}, intersection: {intersection}",
                    }
                )

    save_image(final_image, output_dir / "final_image.png")

    return results


def filled_bottles_pipeline(input_path: str, output_dir: str) -> list[dict]:
    """
    Detects filled bottles in an image.
    """
    image = load_tiff_image(input_path)
    final_image = np.zeros_like(image, dtype=np.uint8)

    histogram_plot = make_histogram(image, log_scale=True)
    save_image(histogram_plot, output_dir / "histogram.png")

    # normalise the image
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # assuming monochromatic background really
    bottle_mask = apply_threshold(image, threshold=5)
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
        bottle_mask = morphological_erode(bottle_mask, kernel_size=5)

        # calculate row means
        row_means = np.ma.array(image, mask=~bottle_mask).mean(axis=1)
        # consider using 50-100 edge pixels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            means_above = np.array(
                [row_means[:i].mean() for i in range(1, len(row_means))]
            )
            means_below = np.array(
                [row_means[i:].mean() for i in range(1, len(row_means))]
            )
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
                **dict(zip(["y", "x"], calculate_centroid(bottle_mask))),
                "liquid_level": liquid_level,
                "shoulder_level": shoulder_level,
                "neck_level": neck_level,
                "is_filled": liquid_level < (shoulder_level + neck_level) / 2,
            }
        )

    # save final image
    save_image(final_image, output_dir / "final_image.png")

    return results
