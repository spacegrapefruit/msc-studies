import matplotlib.pyplot as plt
import numpy as np
import typer
from typing import List

from image_utils import ImageUtils
from image_classes import Image8Bit


app = typer.Typer()
utils = ImageUtils()


@app.command()
def load(input_paths: List[str]):
    for path in input_paths:
        assert path.endswith(".tif")

        image = utils.load_image(path)
        typer.echo(f"Loaded image {path}")
        typer.echo(f"Image shape: {image.shape}")
        typer.echo(f"Image type: {image.dtype}")
        typer.echo("")


@app.command()
def combine(input_paths: List[str], output_path: str = "output.tif"):
    assert len(input_paths) == 3
    assert all(path.endswith(".tif") for path in input_paths)
    assert output_path.endswith(".tif")

    images = [utils.load_image(path) for path in input_paths]

    combined_image = np.stack(images, axis=-1)
    typer.echo(f"Combined image shape: {combined_image.shape}")

    utils.save_image(combined_image, output_path, write_rgb=True)


@app.command()
def power_law(
    input_path: str,
    output_path: str = "output.tif",
    gamma: float = 0.5,
):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    power_law_image = utils.power_law(image, gamma=gamma)
    typer.echo(f"Powerlaw image shape: {power_law_image.shape}")

    utils.save_image(power_law_image, output_path)


@app.command()
def histogram_stretching(input_path: str, output_path: str = "output.tif"):
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    stretched_image = utils.histogram_stretching(image)

    typer.echo(f"Stretched image shape: {stretched_image.shape}")

    utils.save_image(stretched_image, output_path)


@app.command()
def thresholding(
    input_path: str, output_path: str = "output.tif", threshold: int = 127
):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    thresholded_image = utils.thresholding(image, threshold)
    typer.echo(f"Thresholded image shape: {thresholded_image.shape}")

    utils.save_image(thresholded_image, output_path)


@app.command()
def histogram_calculation(input_path: str):
    image = utils.load_image(input_path)

    histogram = utils.calculate_histogram(image)
    typer.echo(f"Histogram shape: {histogram.shape}")
    typer.echo(f"Histogram: {histogram}")

    # plot and save histogram as TIFF
    fig = plt.figure(figsize=(16, 5), dpi=300)
    plt.bar(range(256), histogram)
    plt.xlim([0, 256])
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    fig.savefig("histogram.tif")
    typer.echo("Saved histogram to histogram.tif")


@app.command()
def histogram_normalization(input_path: str, output_path: str = "output.tif"):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    normalized_image = utils.normalize_histogram(image)
    typer.echo(f"Normalized image shape: {normalized_image.shape}")

    utils.save_image(normalized_image, output_path)


@app.command()
def blur(
    input_path: str,
    output_path: str = "output.tif",
    kernel_size: int = 3,
):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    image = Image8Bit(image).to_float()
    image.apply_function(lambda x: utils.averaging_blur(x, kernel_size=kernel_size))
    image = image.to_8bit()
    typer.echo(f"Blurred image shape: {image.shape}")

    image.save(output_path)


@app.command()
def apply_kernel(kernel: str, input_path: str, output_path: str = "output.tif"):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    image = Image8Bit(image).to_float()
    image.apply_function(lambda x: utils.apply_kernel(x, kernel))
    image = image.to_8bit()
    typer.echo(f"Transformed image shape: {image.shape}")

    image.save(output_path)


@app.command()
def sharpen(
    method: str, input_path: str, output_path: str = "output.tif", c: float = 1
):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")
    assert method in ["laplacian", "unsharp"]

    image = utils.load_image(input_path)
    image = Image8Bit(image).to_float()
    if method == "laplacian":
        image.apply_function(lambda x: utils.sharpen_laplacian(x, c=c))
    else:
        image.apply_function(lambda x: utils.sharpen_unsharp(x, c=c))
    image = image.to_8bit()
    typer.echo(f"Sharpened image shape: {image.shape}")

    image.save(output_path)


@app.command()
def sobel(input_path: str, output_path: str = "output.tif"):
    assert input_path.endswith(".tif")
    assert output_path.endswith(".tif")

    image = utils.load_image(input_path)
    image = Image8Bit(image).to_float()
    image.apply_function(lambda x: utils.sobel_operator(x))
    image = image.to_8bit()
    typer.echo(f"Sobel image shape: {image.shape}")

    image.save(output_path)


if __name__ == "__main__":
    app()
