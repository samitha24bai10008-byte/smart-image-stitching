"""
Image preprocessing module.

This module provides utilities for loading, validating,
resizing, and converting images before the stitching process.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given file path.

    Args:
        image_path: Path to the input image.

    Returns:
        Image as a NumPy array in BGR format.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If OpenCV cannot read the image.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image


def resize_image(
    image: np.ndarray,
    max_width: int = 1200,
    max_height: int = 900
) -> np.ndarray:
    """
    Resize an image while maintaining its aspect ratio.

    Images smaller than the specified limits are not enlarged.

    Args:
        image: Input image.
        max_width: Maximum allowed width.
        max_height: Maximum allowed height.

    Returns:
        Resized image.
    """
    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )

    if scale == 1.0:
        return image.copy()

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        image: Input BGR image.

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def validate_image(image: Optional[np.ndarray]) -> bool:
    """
    Validate whether an image is usable.

    Args:
        image: Image represented as a NumPy array.

    Returns:
        True if the image is valid, otherwise False.
    """
    if image is None:
        return False

    if not isinstance(image, np.ndarray):
        return False

    if image.size == 0:
        return False

    if len(image.shape) not in (2, 3):
        return False

    return True


def preprocess_image(
    image_path: str,
    max_width: int = 1200,
    max_height: int = 900
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for one image.

    Args:
        image_path: Path to the input image.
        max_width: Maximum image width.
        max_height: Maximum image height.

    Returns:
        Tuple containing:
            - resized color image
            - grayscale image
    """
    image = load_image(image_path)

    if not validate_image(image):
        raise ValueError(f"Invalid image: {image_path}")

    image = resize_image(
        image,
        max_width=max_width,
        max_height=max_height
    )

    gray_image = convert_to_grayscale(image)

    return image, gray_image