import numpy as np


def _calculate_distance_from_center(shape, relative_center=None):
    M, N = shape
    y, x = np.ogrid[:M, :N]
    if relative_center is None:
        center_y, center_x = M // 2, N // 2
    else:
        center_y = M // 2 + relative_center[0]
        center_x = N // 2 + relative_center[1]
    return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)


## Part 1.3 - Frequency Domain Filters


def ideal_low_pass_filter(shape, cutoff):
    distance_from_center = _calculate_distance_from_center(shape)
    filter_mask = (distance_from_center <= cutoff).astype(float)
    return filter_mask


def ideal_high_pass_filter(shape, cutoff):
    return 1 - ideal_low_pass_filter(shape, cutoff)


def butterworth_low_pass_filter(shape, cutoff, order=2):
    distance_from_center = _calculate_distance_from_center(shape)
    filter_mask = 1 / (1 + (distance_from_center / cutoff) ** (2 * order))
    return filter_mask


def butterworth_high_pass_filter(shape, cutoff, order=2):
    return 1 - butterworth_low_pass_filter(shape, cutoff, order)


def gaussian_low_pass_filter(shape, cutoff):
    distance_from_center = _calculate_distance_from_center(shape)
    filter_mask = np.exp(-((distance_from_center) ** 2) / (2 * (cutoff**2)))
    return filter_mask


def gaussian_high_pass_filter(shape, cutoff):
    return 1 - gaussian_low_pass_filter(shape, cutoff)


## Part 1.3 - Linear Filters in the Frequency Domain


def average_kernel(size):
    return np.ones((size, size), dtype=float) / (size**2)


def differentiation_x_kernel():
    return np.array([[-1, 0, 1]], dtype=float)


def gaussian_kernel(size, sigma):
    distance_from_center = _calculate_distance_from_center((size, size))
    filter_mask = np.exp(-((distance_from_center) ** 2) / (2 * (sigma**2)))
    filter_mask /= filter_mask.sum()
    return filter_mask


def to_frequency_domain(kernel_mask):
    def spatial_kernel_filter(shape):
        M, N = shape
        padded_filter = np.zeros(shape, dtype=float)
        padded_filter[: kernel_mask.shape[0], : kernel_mask.shape[1]] = kernel_mask

        y, x = np.ogrid[:M, :N]
        padded_filter = padded_filter * (-1) ** (x + y)
        dft_kernel = np.abs(np.fft.fft2(padded_filter))

        return dft_kernel

    return spatial_kernel_filter


## Part 1.4 - Noise Removal Filters


def ideal_band_pass_filter(shape, center, bandwidth):
    filter_mask = ideal_low_pass_filter(shape, center + bandwidth / 2)
    filter_mask -= ideal_low_pass_filter(shape, center - bandwidth / 2)
    return filter_mask


def ideal_band_reject_filter(shape, center, bandwidth):
    return 1 - ideal_band_pass_filter(shape, center, bandwidth)


def gaussian_band_pass_filter(shape, center, bandwidth):
    distance_from_center = _calculate_distance_from_center(shape)

    # Gaussian band-pass formula
    filter_mask = np.exp(-((distance_from_center - center) ** 2) / (2 * (bandwidth**2)))

    return filter_mask


def gaussian_band_reject_filter(shape, center, bandwidth):
    return 1 - gaussian_band_pass_filter(shape, center, bandwidth)


def butterworth_band_pass_filter(shape, center, bandwidth, order=2):
    distance_from_center = _calculate_distance_from_center(shape)

    # Butterworth band-pass formula
    filter_mask = 1 / (
        1
        + ((distance_from_center * bandwidth) / (distance_from_center**2 - center**2))
        ** (2 * order)
    )

    return filter_mask


def butterworth_band_reject_filter(shape, center, bandwidth, order=2):
    return 1 - butterworth_band_pass_filter(shape, center, bandwidth, order)


def ideal_notch_filter(shape, centers, radius):
    M, N = shape
    filter_mask = np.ones((M, N))
    for center in centers:
        distance_from_center = _calculate_distance_from_center(shape, center)
        mask = distance_from_center > radius
        filter_mask *= mask
        distance_from_center = _calculate_distance_from_center(
            shape, (-center[0], -center[1])
        )
        mask = distance_from_center > radius
        filter_mask *= mask
    return filter_mask


def gaussian_notch_filter(shape, centers, radius):
    M, N = shape
    filter_mask = np.ones((M, N))
    for center in centers:
        distance_from_center = _calculate_distance_from_center(shape, center)
        mask = 1 - np.exp(-(distance_from_center**2) / (2 * (radius**2)))
        filter_mask *= mask
        distance_from_center = _calculate_distance_from_center(
            shape, (-center[0], -center[1])
        )
        mask = 1 - np.exp(-(distance_from_center**2) / (2 * (radius**2)))
        filter_mask *= mask
    return filter_mask


# TODO allow order parameter
# def butterworth_notch_filter(shape, centers, radius, order=2):
#     M, N = shape
#     filter_mask = np.ones((M, N))
#     for center in centers:
#         distance_from_center = _calculate_distance_from_center(shape, center)
#         mask = 1 / (1 + (distance_from_center / radius) ** (2 * order))
#         filter_mask *= mask
#     return filter_mask


def median_filter(image, kernel_size=3):
    M, N = image.shape
    padded_image = np.pad(image, pad_width=kernel_size // 2, mode="edge")
    filtered_image = np.zeros_like(image)
    for i in range(M):
        for j in range(N):
            neighborhood = padded_image[i : i + kernel_size, j : j + kernel_size]
            filtered_image[i, j] = np.median(neighborhood)
    return filtered_image


def mean_filter(image, kernel_size=3):
    M, N = image.shape
    padded_image = np.pad(image, pad_width=kernel_size // 2, mode="edge")
    filtered_image = np.zeros_like(image)
    for i in range(M):
        for j in range(N):
            neighborhood = padded_image[i : i + kernel_size, j : j + kernel_size]
            filtered_image[i, j] = np.mean(neighborhood)
    return filtered_image


def mean_squared_error(original, restored):
    return np.mean((original - restored) ** 2)


def apply_adaptive_filter(image, filter_type="median"):
    if filter_type == "median":
        filtered_image = median_filter(image, kernel_size=3)
    elif filter_type == "mean":
        filtered_image = mean_filter(image, kernel_size=3)
    return filtered_image


kernel_map = {
    "average": average_kernel,
    "differentiation_x": differentiation_x_kernel,
    "gaussian": gaussian_kernel,
}


filter_map = {
    "butterworth_band_pass": butterworth_band_pass_filter,
    "butterworth_band_reject": butterworth_band_reject_filter,
    "butterworth_low_pass": butterworth_low_pass_filter,
    "butterworth_high_pass": butterworth_high_pass_filter,
    # "butterworth_notch": butterworth_notch_filter,
    "gaussian_band_pass": gaussian_band_pass_filter,
    "gaussian_band_reject": gaussian_band_reject_filter,
    "gaussian_low_pass": gaussian_low_pass_filter,
    "gaussian_high_pass": gaussian_high_pass_filter,
    "gaussian_notch": gaussian_notch_filter,
    "ideal_band_pass": ideal_band_pass_filter,
    "ideal_band_reject": ideal_band_reject_filter,
    "ideal_low_pass": ideal_low_pass_filter,
    "ideal_high_pass": ideal_high_pass_filter,
    "ideal_notch": ideal_notch_filter,
}
