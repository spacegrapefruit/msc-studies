import logging
from enum import Enum

import numpy as np
from image_utils import (
    apply_threshold,
    calculate_centroid,
    calculate_dimensions,
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
    # is_centered,
)


class Colour(Enum):
    FUSE_WIRE = 16
    SOLDER = 64
    SOLDERED_CONNECTION = 80
    SILICON = 128
    MICROCONTROLLER = 176
    HOLES = 240
    # 80
    # 96


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

    # morphological erosion + dilation of cells
    dapi_mask = morphological_erode(dapi_mask, kernel_size=5)
    dapi_mask = morphological_dilate(dapi_mask, kernel_size=5)

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


def circuit_board_qa_pipeline(input_path: str, output_dir: str):
    image = load_tiff_image(input_path)

    # fix salt-and-pepper noise using median filtering
    noise_mask = (image == 0) | (image == 255)
    image_filtered = median_filter(image, kernel_size=1)
    image[noise_mask] = image_filtered[noise_mask]

    save_image(image, output_dir / "median_filtered_image.png")

    image[image == Colour.SOLDERED_CONNECTION.value] = Colour.SOLDER.value

    holes_mask = image == Colour.HOLES.value
    conductor_mask = (image == Colour.SOLDER.value) | holes_mask

    conductor_mask_thick = morphological_erode(conductor_mask, kernel_size=5)
    conductor_mask_thick = (
        morphological_dilate(conductor_mask_thick, kernel_size=5) & conductor_mask
    )
    conductor_image = (conductor_mask_thick * 255).astype(np.uint8)
    save_image(conductor_image, output_dir / "conductor_mask_thick.png")

    labeled_components, num_components = label_connected_components(conductor_mask)
    labeled_holes, num_holes = label_connected_components(holes_mask)
    holes_checked = np.zeros(num_holes + 1, dtype=bool)

    # FIXME naming
    final_image = np.full((*image.shape, 3), Colour.SILICON.value, dtype=np.uint8)
    for component_idx in range(1, num_components + 1):
        component_mask = labeled_components == component_idx

        this_component_mask_thick = component_mask & conductor_mask_thick
        # this_holes_mask = holes_mask & component_mask

        labeled_conductors, num_conductors = label_connected_components(
            this_component_mask_thick
        )

        has_wires = (component_mask & (~this_component_mask_thick)).sum() > 0

        final_image[component_mask, :] = Colour.SOLDER.value + 16  # make wires brighter
        if has_wires and num_conductors < 2:
            final_image[component_mask, 1:3] = 0  # set green, blue channels to 0
            logging.warning(
                f"Component {component_idx} with suspected broken wire touches {num_conductors} connectors"
            )

        final_image[this_component_mask_thick] = Colour.SOLDER.value
        # TODO check soldering regions
        for conductor_idx in range(1, num_conductors + 1):
            this_this_conductor_mask = labeled_conductors == conductor_idx
            dims = calculate_dimensions(this_this_conductor_mask)
            min_dim, max_dim = sorted(dims)
            centroid = calculate_centroid(this_this_conductor_mask)
            area = np.sum(this_this_conductor_mask)
            fill_frac = area / (min_dim * max_dim)

            if min_dim == 6 and max_dim == 9:
                # component connector
                continue
            elif min_dim == 7 and max_dim in (11, 12, 13):
                # contact connector
                continue
            elif min_dim == 17 and max_dim in (17, 20):
                # soldering region

                # TODO move to a function
                expected_mask = np.zeros_like(this_this_conductor_mask)

                if fill_frac < 0.9:
                    shape = "round"
                    expected_r = (241 / np.pi) ** 0.5
                    expected_mask[
                        np.linalg.norm(
                            np.indices(expected_mask.shape)
                            - np.expand_dims(centroid, axis=[1, 2]),
                            axis=0,
                        )
                        < expected_r
                    ] = 1
                else:
                    shape = "square"
                    expected_w, expected_h = (20, 17)
                    y_mask = (
                        np.abs(np.arange(expected_mask.shape[0]) - centroid[0])
                        < expected_h / 2
                    )
                    x_mask = (
                        np.abs(np.arange(expected_mask.shape[1]) - centroid[1])
                        < expected_w / 2
                    )
                    expected_mask[np.ix_(y_mask, x_mask)] = 1

                intersection = np.sum(this_this_conductor_mask & expected_mask)

                if intersection < max(area, expected_mask.sum()) * 0.99:
                    final_image[this_this_conductor_mask, 1:3] = 0
                    logging.warning(
                        f"Component {component_idx}, {shape} soldering region {conductor_idx}, area: {area}, expected: {expected_mask.sum()}, intersection: {intersection}"
                    )

                holes_in_conductor = sorted(
                    set(np.unique(labeled_holes[this_this_conductor_mask])) - {0}
                )
                holes_checked[holes_in_conductor] = True
                if len(holes_in_conductor) != 1:
                    logging.warning(
                        f"Component {component_idx}, {shape} soldering region {conductor_idx} has holes: {holes_in_conductor}"
                    )
                else:
                    hole_mask = labeled_holes == holes_in_conductor[0]
                    final_image[hole_mask] = Colour.HOLES.value
                    hole_centroid = calculate_centroid(hole_mask)
                    hole_dims = calculate_dimensions(hole_mask)
                    if hole_centroid != centroid:
                        logging.warning(
                            f"Component {component_idx}, {shape} soldering region {conductor_idx} centroid mismatch: {centroid} != {hole_centroid}"
                        )
                        final_image[hole_mask, 1:3] = 0
            else:
                # catch and add to connector / soldering region
                raise ValueError(f"Unknown region dimensions: {dims}")

    save_image(final_image, output_dir / "final_image.png")


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
        bottle_mask = morphological_erode(bottle_mask, kernel_size=5)

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
