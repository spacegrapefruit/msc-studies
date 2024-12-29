import logging
import numpy as np
from libtiff import TIFF
from matplotlib import pyplot as plt


EPSILON = 1e-6


def read_image(file_path):
    tif = TIFF.open(file_path, mode="r")
    image = tif.read_image()
    tif.close()

    # normalize the image to [0, 1]
    image = image.astype(float) / 255
    return image


def save_image(image, file_base_path, extensions):
    assert image.dtype in [np.float64, np.complex128]

    if image.dtype == np.complex128:
        logging.warning(
            "Encountered complex image. Saving log(abs(image) + 1) instead."
        )
        image = np.log(np.abs(image) + 1)
        image = image / image.max()
        image = (image * 255).astype(np.uint8)

    elif image.dtype == np.float64:
        image = image.round(
            6
        )  # round to 6 decimal places to prevent warnings due to FP precision
        if np.any(image < 0) or np.any(image > 1):
            logging.warning(
                "Image values are outside the valid range [0, 1]. Clipping to [0, 1]"
            )
            image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = image.astype(np.uint8)

    for extension in extensions:
        file_path = f"{file_base_path}{extension}"

        if extension == ".png":
            plt.imsave(file_path, image, cmap="gray", vmin=0, vmax=255)
        elif extension == ".tif":
            tif = TIFF.open(file_path, mode="w")
            tif.write_image(image)
            tif.close()
        logging.info(f"Saved image to {file_path}")


def pad_image(image):
    M, N = image.shape
    padded_image = np.pad(image, ((0, M), (0, N)), mode="constant")
    return padded_image


def shift_image(image):
    M, N = image.shape
    y, x = np.ogrid[:M, :N]

    # shift the image to center the frequencies
    shifted_image = image * (-1) ** (x + y)
    return shifted_image


def extract_upper_left(image):
    M, N = image.shape
    # extract the upper-left quadrant of the image
    cropped_image = image[: M // 2, : N // 2]
    return cropped_image


def generate_image(
    shape,
    method,
    size,
):
    M, N = shape
    alpha = 2.0 * np.pi / size

    y, x = np.ogrid[:M, :N]

    if method == "sine":
        image = np.cos(alpha * x) * np.cos(alpha * y)
        image = (image + 1) / 2  # squeeze to [0, 1]
    elif method == "vertical_lines":
        image = np.sign(np.sin(alpha * x + EPSILON)) * np.ones_like(
            y
        )  # add epsilon to avoid sign(0)
        image = (image + 1) / 2  # squeeze to [0, 1]
    elif method == "checkerboard":
        image = np.sign(np.sin(alpha * x + EPSILON) * np.sin(alpha * y + EPSILON))
        image = (image + 1) / 2  # squeeze to [0, 1]
    else:
        raise ValueError(f"Invalid method: {method}")

    return image


def convolve(image, kernel):
    K, L = kernel.shape
    M, N = image.shape
    pad_height = K // 2
    pad_width = L // 2

    padded_image = np.pad(
        image,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
    )
    convolved_image = np.zeros_like(image)

    for i in range(M):
        for j in range(N):
            region = padded_image[i : i + K, j : j + L]
            convolved_image[i, j] = np.sum(region * kernel)

    return convolved_image


def apply_filter(dft_image, filter_func):
    filter_shape = dft_image.shape
    filter_mask = filter_func(shape=filter_shape)
    return {
        "image": dft_image * filter_mask,
        "mask": filter_mask,
    }
