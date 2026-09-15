"""
Image warping module.

This module calculates the required panorama canvas,
warps an image using a homography matrix, and combines
it with the reference image.
"""

import cv2
import numpy as np


def get_image_corners(
    image: np.ndarray
) -> np.ndarray:
    """
    Get the four corner coordinates of an image.

    Args:
        image: Input image.

    Returns:
        Array containing four corner points.
    """
    height, width = image.shape[:2]

    return np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])


def transform_corners(
    image: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    Transform image corners using a homography matrix.

    Args:
        image: Input image.
        homography: 3x3 homography matrix.

    Returns:
        Transformed corner coordinates.
    """
    corners = get_image_corners(image)

    homogeneous_corners = np.column_stack([
        corners,
        np.ones(len(corners))
    ])

    transformed = (
        homography @ homogeneous_corners.T
    ).T

    transformed /= transformed[:, 2:3]

    return transformed[:, :2]


def calculate_canvas(
    image1: np.ndarray,
    image2: np.ndarray,
    homography: np.ndarray
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Calculate the size and translation matrix of the panorama canvas.

    Image 1 is transformed according to the homography and
    image 2 remains in its original coordinate system.

    Args:
        image1: Source image.
        image2: Reference image.
        homography: Homography mapping image 1 to image 2.

    Returns:
        Tuple containing:
            - canvas size as (width, height)
            - translation matrix.
    """
    corners1 = transform_corners(
        image1,
        homography
    )

    corners2 = get_image_corners(image2)

    all_corners = np.vstack([
        corners1,
        corners2
    ])

    min_x = int(np.floor(np.min(all_corners[:, 0])))
    min_y = int(np.floor(np.min(all_corners[:, 1])))

    max_x = int(np.ceil(np.max(all_corners[:, 0])))
    max_y = int(np.ceil(np.max(all_corners[:, 1])))

    translation_x = -min_x if min_x < 0 else 0
    translation_y = -min_y if min_y < 0 else 0

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(
            "Invalid panorama canvas dimensions."
        )

    translation_matrix = np.array([
        [1.0, 0.0, translation_x],
        [0.0, 1.0, translation_y],
        [0.0, 0.0, 1.0]
    ])

    return (
        (canvas_width, canvas_height),
        translation_matrix
    )


def warp_image(
    image: np.ndarray,
    homography: np.ndarray,
    canvas_size: tuple[int, int],
    translation_matrix: np.ndarray
) -> np.ndarray:
    """
    Warp an image onto the panorama canvas.

    Args:
        image: Image to warp.
        homography: Homography matrix.
        canvas_size: Canvas dimensions (width, height).
        translation_matrix: Translation matrix.

    Returns:
        Warped image.
    """
    combined_transform = (
        translation_matrix @ homography
    )

    return cv2.warpPerspective(
        image,
        combined_transform,
        canvas_size
    )


def place_reference_image(
    image: np.ndarray,
    canvas_size: tuple[int, int],
    translation_matrix: np.ndarray
) -> np.ndarray:
    """
    Place the reference image onto the panorama canvas.

    Args:
        image: Reference image.
        canvas_size: Canvas dimensions (width, height).
        translation_matrix: Translation matrix.

    Returns:
        Reference image placed on the canvas.
    """
    return cv2.warpPerspective(
        image,
        translation_matrix,
        canvas_size
    )


def create_warped_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    homography: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp both images onto a common panorama canvas.

    Args:
        image1: First image.
        image2: Second/reference image.
        homography: Homography mapping image 1 to image 2.

    Returns:
        Tuple containing:
            - warped first image
            - placed second image
    """
    canvas_size, translation_matrix = calculate_canvas(
        image1,
        image2,
        homography
    )

    warped_image1 = warp_image(
        image1,
        homography,
        canvas_size,
        translation_matrix
    )

    placed_image2 = place_reference_image(
        image2,
        canvas_size,
        translation_matrix
    )

    return warped_image1, placed_image2