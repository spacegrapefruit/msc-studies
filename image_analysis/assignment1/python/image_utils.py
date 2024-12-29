import typer
import numpy as np
from libtiff import TIFF


def averaging_blur_kernel(size):
    return np.ones((size, size)) / (size**2)


kernels = {
    "derivative_x": np.array([[0, -1, 1]]),
    "derivative_y": np.array([[0], [-1], [1]]),
    "second_derivative_x": np.array([[1, -2, 1]]),
    "second_derivative_y": np.array([[1], [-2], [1]]),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
}


class ImageUtils:
    def load_image(self, path: str) -> np.ndarray:
        image = TIFF.open(path).read_image()
        return image

    def save_image(self, image: np.ndarray, path: str, **kwargs):
        tif = TIFF.open(path, mode="w")
        tif.write_image(image, **kwargs)
        tif.close()
        typer.echo(f"Saved transformed image to {path}")

    # part 2
    def power_law(self, image: np.ndarray, gamma: float) -> np.ndarray:
        normalized_image = image / 255.0
        transformed_image = np.power(normalized_image, gamma) * 255
        return transformed_image.astype(np.uint8)

    def histogram_stretching(self, image: np.ndarray) -> np.ndarray:
        target_min_value = 0
        target_max_value = 255

        min_value = image.min()
        max_value = image.max()
        typer.echo(f"Min value: {min_value}")
        typer.echo(f"Max value: {max_value}")

        return (image - min_value) / (max_value - min_value) * (
            target_max_value - target_min_value
        ) + target_min_value

    def thresholding(self, image: np.ndarray, threshold: int) -> np.ndarray:
        return (image > threshold) * 255

    def calculate_histogram(self, image: np.ndarray) -> np.ndarray:
        histogram, _ = np.histogram(image, bins=256, range=(0, 255))
        return histogram

    def normalize_histogram(self, image: np.ndarray) -> np.ndarray:
        histogram = self.calculate_histogram(image)
        cdf = histogram.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        image_equalized = np.interp(image.flatten(), range(0, 256), cdf_normalized)
        return image_equalized.reshape(image.shape).astype(np.uint8)

    ## part 3
    def convolve(self, image, kernel):
        kernel_height, kernel_width = kernel.shape
        image_height, image_width = image.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded_image = np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width)),
            mode="constant",
        )
        convolved_image = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i : i + kernel_height, j : j + kernel_width]
                convolved_image[i, j] = np.sum(region * kernel)

        return convolved_image

    def averaging_blur(self, image, kernel_size):
        kernel = averaging_blur_kernel(kernel_size)
        blurred_image = self.convolve(image.data, kernel)
        return blurred_image

    def apply_kernel(self, image, kernel):
        transformed_image = self.convolve(image, kernels[kernel])
        return transformed_image

    def sharpen_laplacian(self, image, c=1):
        sharpened_image = image + self.apply_kernel(image, "laplacian") * c
        return sharpened_image

    def sharpen_unsharp(self, image, c=1):
        blurred_image = self.averaging_blur(image, kernel_size=3)
        mask = image - blurred_image
        sharpened_image = image + c * mask
        return sharpened_image

    def sobel_operator(self, image):
        dx = self.convolve(image.data, kernels["sobel_x"])
        dy = self.convolve(image.data, kernels["sobel_y"])
        magnitude = np.sqrt(dx**2 + dy**2)
        return magnitude

    def multistep_processing(self, image):
        raise NotImplementedError


## part 4
def index_to_local(index, pixel_size):
    return index * pixel_size


def local_to_world(local, origin):
    return local + origin


def world_to_local(world, origin):
    return world - origin


def local_to_index(local, pixel_size):
    return local / pixel_size


def create_affine_matrix(scale=(1, 1), rotation=0, translation=(0, 0)):
    scale_matrix = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])

    rotation_matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1],
        ]
    )

    translation_matrix = np.array(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]
    )

    affine_matrix = translation_matrix @ rotation_matrix @ scale_matrix
    return affine_matrix


def apply_affine_transform(image, affine_matrix, output_shape, interpolation="nearest"):
    inverse_affine_matrix = np.linalg.inv(affine_matrix)
    output_image = np.zeros(output_shape, dtype=image.dtype)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            src_coords = inverse_affine_matrix @ np.array([i, j, 1])
            src_x, src_y = src_coords[:2]

            if interpolation == "nearest":
                src_x = int(round(src_x))
                src_y = int(round(src_y))
                if 0 <= src_x < image.shape[0] and 0 <= src_y < image.shape[1]:
                    output_image[i, j] = image[src_x, src_y]
            elif interpolation == "bilinear":
                x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                x1, y1 = x0 + 1, y0 + 1

                if 0 <= x0 < image.shape[0] and 0 <= y0 < image.shape[1]:
                    Ia = image[x0, y0]
                else:
                    Ia = 0

                if 0 <= x1 < image.shape[0] and 0 <= y1 < image.shape[1]:
                    Ib = image[x1, y0]
                else:
                    Ib = 0

                if 0 <= x0 < image.shape[0] and 0 <= y1 < image.shape[1]:
                    Ic = image[x0, y1]
                else:
                    Ic = 0

                if 0 <= x1 < image.shape[0] and 0 <= y1 < image.shape[1]:
                    Id = image[x1, y1]
                else:
                    Id = 0

                wa = (x1 - src_x) * (y1 - src_y)
                wb = (src_x - x0) * (y1 - src_y)
                wc = (x1 - src_x) * (src_y - y0)
                wd = (src_x - x0) * (src_y - y0)

                output_image[i, j] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return output_image
