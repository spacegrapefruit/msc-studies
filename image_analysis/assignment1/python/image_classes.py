from abc import ABC

import numpy as np
import typer
from libtiff import TIFF


class ImageABC(ABC):
    def __init__(self, data):
        self._data = data

    def apply_function(self, function):
        self._data = function(self._data)

    @property
    def shape(self):
        return self._data.shape


class Image8Bit(ImageABC):
    def __init__(self, data):
        assert data.dtype == np.uint8

        super().__init__(data)

    def to_float(self):
        return ImageFloat(self._data.astype(np.float32) / 255.0)

    def save(self, path, **kwargs):
        tif = TIFF.open(path, mode="w")
        tif.write_image(self._data, **kwargs)
        tif.close()
        typer.echo(f"Saved image to {path}")


class ImageFloat(ImageABC):
    def __init__(self, data):
        assert data.dtype == np.float32

        super().__init__(data)

    def to_8bit(self):
        clipped_data = np.clip(self._data * 255.0, 0, 255).astype(np.uint8)
        return Image8Bit(clipped_data)
