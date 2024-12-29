import logging
import typer
from typing import List, Optional, Tuple

from image_pipelines import (
    ConvolutionPipeline,
    FrequencyFilteringPipeline,
    ImageGenerationFourierTransformPipeline,
    PeriodicNoiseAnalysisPipeline,
    PeriodicNoiseRemovalPipeline,
    TwoWayFourierTransformPipeline,
)


app = typer.Typer()


@app.command()
def two_way_fourier(input_path: str):
    pipeline = TwoWayFourierTransformPipeline()
    pipeline.process(input_path)


@app.command()
def image_generation(
    method: str,
    shape: Tuple[int, int] = (256, 256),
    size: int = 4,
):
    pipeline = ImageGenerationFourierTransformPipeline(
        method=method,
        shape=shape,
        size=size,
    )
    pipeline.process()


@app.command()
def convolution(
    kernel_name: str,
    input_path: str,
    size: Optional[int] = None,
    sigma: Optional[float] = None,
):
    pipeline = ConvolutionPipeline(
        kernel_name=kernel_name,
        size=size,
        sigma=sigma,
    )
    pipeline.process(input_path)


@app.command()
def frequency_filtering(
    filter_name: str,
    input_path: str,
    cutoff: Optional[int] = None,
    order: Optional[int] = None,
    sigma: Optional[float] = None,
    size: Optional[int] = None,
):
    pipeline = FrequencyFilteringPipeline(
        filter_name=filter_name,
        cutoff=cutoff,
        order=order,
        sigma=sigma,
        size=size,
    )
    pipeline.process(input_path)


@app.command()
def noise_analysis(
    input_path: str,
    original_image_path: str,
):
    pipeline = PeriodicNoiseAnalysisPipeline(
        original_image_path=original_image_path,
    )
    pipeline.process(input_path)


@app.command()
def noise_removal(
    filter_name: str,
    input_path: str,
    center: Optional[int] = None,
    bandwidth: Optional[int] = None,
    centers: Optional[
        str
    ] = None,  # sadly, typer does not support List[Tuple[int, int]]
    radius: Optional[int] = None,
):
    if centers is not None:
        centers = [tuple(map(int, c.split(","))) for c in centers.split("/")]
        assert len(centers) > 0
        assert all(len(center) == 2 for center in centers)

    pipeline = PeriodicNoiseRemovalPipeline(
        filter_name=filter_name,
        center=center,
        bandwidth=bandwidth,
        centers=centers,
        radius=radius,
    )
    pipeline.process(input_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app()
