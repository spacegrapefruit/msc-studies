import numpy as np
import typer
from pathlib import Path
from functools import partial

from image_utils import (
    apply_filter,
    convolve,
    extract_upper_left,
    generate_image,
    pad_image,
    read_image,
    save_image,
    shift_image,
)
from filters import filter_map, kernel_map, to_frequency_domain

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"


class ImageProcessingPipeline:
    output_dir = None

    def process(self, path=None):
        output_dir = self.output_dir / Path(path).stem
        output_dir.mkdir(parents=True, exist_ok=True)

        image = path  # ugly code
        results = []
        for i, (transform_name, transform) in enumerate(self.transforms, start=1):
            image = transform(image)

            mask = None
            if isinstance(image, dict):
                mask = image["mask"]
                image = image["image"]

            results.append((transform_name, image))

            typer.echo(f"Transform {i}: {transform_name}")
            typer.echo(f"Image shape: {image.shape}")
            typer.echo(f"Image type: {image.dtype}")
            typer.echo(
                f"Image min: {image.min().round(3)}, mean: {image.mean().round(3)}, max: {image.max().round(3)}"
            )

            filename = f"{i:02d}_{transform_name.replace(' ', '_').lower()}"
            output_path = output_dir / filename
            save_image(image, output_path, extensions=[".png", ".tif"])
            if mask is not None:
                mask_path = output_dir / f"{filename}_mask"
                save_image(mask, mask_path, extensions=[".png"])

            typer.echo("")
        return results


class TwoWayFourierTransformPipeline(ImageProcessingPipeline):
    output_subdir = "two_way_fourier"

    def __init__(self):
        self.output_dir = OUTPUT_DIR / self.output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transforms = [
            # step 1: read the input image
            ("Original Image", read_image),
            # step 2: pad the image to 2M x 2N
            ("Padded Image", pad_image),
            # step 3: shift the padded image for periodicity
            ("Shifted for Periodicity", shift_image),
            # step 4: compute DFT
            ("DFT", np.fft.fft2),
            # step 5: compute the IDFT of the DFT image
            ("Inverse DFT", np.fft.ifft2),
            # step 6: calculate the absolute value of the IDFT
            ("Magnitude of IDFT", np.abs),
            # step 7: extract the upper-left quadrant of the image
            ("Upper Left Quadrant", extract_upper_left),
        ]


class ImageGenerationFourierTransformPipeline(ImageProcessingPipeline):
    output_subdir = "image_generation"

    def __init__(
        self,
        method,
        shape,
        size,
    ):
        self.output_dir = OUTPUT_DIR / self.output_subdir / method
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.generate_image = lambda _: generate_image(
            shape,
            method=method,
            size=size,
        )

        self.transforms = [
            # step 1: generate an image using the specified method
            ("Generate Image", self.generate_image),
            # step 2: compute DFT
            ("DFT", np.fft.fft2),
            # step 3: shift the DFT image for visualization
            ("Shifted DFT", np.fft.fftshift),
        ]


class ConvolutionPipeline(ImageProcessingPipeline):
    def __init__(
        self,
        kernel_name,
        size=None,
        sigma=None,
    ):
        self.output_dir = OUTPUT_DIR / "convolution" / kernel_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        kernel_func = kernel_map[kernel_name]
        if kernel_name == "average":
            kernel_mask = kernel_func(size=size)
        elif kernel_name == "gaussian":
            kernel_mask = kernel_func(size=size, sigma=sigma)
        elif kernel_name == "differentiation_x":
            kernel_mask = kernel_func()

        self.transforms = [
            # step 1: read the input image
            ("Original Image", read_image),
            # step 2: apply the convolution kernel
            ("Convolved Image", lambda image: convolve(image, kernel_mask)),
        ]


class FrequencyFilteringPipeline(ImageProcessingPipeline):
    def __init__(
        self,
        filter_name,
        cutoff,
        order,
        sigma,
        size,
    ):
        self.output_dir = OUTPUT_DIR / "frequency_filtering" / filter_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filter_name.startswith("spatial"):
            kernel_func = kernel_map[filter_name.replace("spatial_", "")]
            if filter_name == "spatial_average":
                kernel_mask = kernel_func(size=size)
            elif filter_name == "spatial_gaussian":
                kernel_mask = kernel_func(size=size, sigma=sigma)
            elif filter_name == "spatial_differentiation_x":
                kernel_mask = kernel_func()
            partial_filter = to_frequency_domain(kernel_mask)
        else:
            filter_func = filter_map[filter_name]
            if filter_name.startswith("ideal"):
                partial_filter = partial(filter_func, cutoff=cutoff)
            elif filter_name.startswith("butterworth"):
                partial_filter = partial(filter_func, cutoff=cutoff, order=order)
            elif filter_name.startswith("gaussian"):
                partial_filter = partial(filter_func, cutoff=cutoff)

        self.transforms = [
            # step 1: read the input image
            ("Original Image", read_image),
            # step 2: pad the image to 2M x 2N
            ("Padded Image", pad_image),
            # step 3: shift the padded image for periodicity
            ("Shifted for Periodicity", shift_image),
            # step 4: compute DFT
            ("DFT", np.fft.fft2),
            # step 5: apply the chosen filter
            (
                "Applied Filter",
                lambda dft_image: apply_filter(dft_image, partial_filter),
            ),
            # step 6: compute the IDFT of the DFT image
            ("Inverse DFT", np.fft.ifft2),
            # step 7: calculate the absolute value of the IDFT
            ("Magnitude of IDFT", np.abs),
            # step 8: extract the upper-left quadrant of the image
            ("Upper Left Quadrant", extract_upper_left),
        ]


class PeriodicNoiseAnalysisPipeline(ImageProcessingPipeline):
    def __init__(self, original_image_path):
        self.output_dir = OUTPUT_DIR / "periodic_noise_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transforms = [
            # step 1: read the input image
            ("Noisy Image", read_image),
            # step 2: subtract the original image from the noisy image
            (
                "Difference Image",
                lambda noisy_image: noisy_image - read_image(original_image_path),
            ),
            # step 3: pad the image to 2M x 2N
            ("Padded Image", pad_image),
            # step 4: shift the padded image for periodicity
            ("Shifted for Periodicity", shift_image),
            # step 5: compute DFT
            ("DFT", np.fft.fft2),
        ]


class PeriodicNoiseRemovalPipeline(ImageProcessingPipeline):
    def __init__(
        self,
        filter_name,
        center=None,
        bandwidth=None,
        centers=None,
        radius=None,
    ):
        self.output_dir = OUTPUT_DIR / "periodic_noise_removal" / filter_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filter_func = filter_map[filter_name]
        if filter_name.startswith("ideal_band"):
            assert center is not None
            assert bandwidth is not None
            partial_filter = partial(filter_func, center=center, bandwidth=bandwidth)
        elif filter_name.startswith("gaussian_band"):
            assert center is not None
            assert bandwidth is not None
            partial_filter = partial(filter_func, center=center, bandwidth=bandwidth)
        elif filter_name == "ideal_notch":
            assert centers is not None
            assert radius is not None
            partial_filter = partial(filter_func, centers=centers, radius=radius)
        elif filter_name == "gaussian_notch":
            assert centers is not None
            assert radius is not None
            partial_filter = partial(filter_func, centers=centers, radius=radius)

        self.transforms = [
            # step 1: read the input image
            ("Original Image", read_image),
            # step 2: pad the image to 2M x 2N
            ("Padded Image", pad_image),
            # step 3: shift the padded image for periodicity
            ("Shifted for Periodicity", shift_image),
            # step 4: compute DFT
            ("DFT", np.fft.fft2),
            # step 5: apply the chosen filter
            (
                "Applied Filter",
                lambda dft_image: apply_filter(dft_image, partial_filter),
            ),
            # step 6: compute the IDFT of the DFT image
            ("Inverse DFT", np.fft.ifft2),
            # step 7: calculate the absolute value of the IDFT
            ("Magnitude of IDFT", np.abs),
            # step 8: extract the upper-left quadrant of the image
            ("Upper Left Quadrant", extract_upper_left),
        ]
