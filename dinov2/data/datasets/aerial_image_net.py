from typing import Any, Tuple, Callable, Optional

import os
import numpy as np
import rasterio

from dinov2.data.datasets.extended import ExtendedVisionDataset

class AerialImageNet(ExtendedVisionDataset):
    """Aerial Image Dataset for Self-Supervised Learning using rasterio.

    Args:
        root (str): Root directory of the dataset containing .tif files.
        transforms (Optional[Callable], optional): A function/transform that takes in
            a 4-channel aerial image and returns a transformed version.
        bands (Tuple[int, int, int, int]): Indices of the bands to load (default is R, G, B, NIR).
    """
    def __init__(
        self, 
        root: str, 
        transforms: Optional[Callable] = None, 
        bands: Tuple[int, int, int, int] = (0, 1, 2, 3),  
        **kwargs 
    ) -> None:
        super().__init__(root, transforms=transforms, **kwargs)
        self.bands = bands
        self.image_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".tif")]

    def get_image_data(self, index: int) -> np.ndarray:
        """Loads a 4-channel aerial image chip.

        Args:
            index (int): Index of the image chip to load.

        Returns:
            np.ndarray: 4-channel aerial image chip.
        """
        filepath = self.image_files[index]

        with rasterio.open(filepath) as src:
            image_data = src.read(self.bands)
        return image_data.transpose(1, 2, 0)  # (H, W, C) 

    def get_target(self, index: int) -> Any:
        return None  # No targets for self-supervised learning

    def __len__(self) -> int:
        return len(self.image_files)