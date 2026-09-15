"""
Image blending module.

Combines two warped images into a single panorama and
smoothly blends their overlapping regions.
"""

import cv2
import numpy as np


def create_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask representing valid pixels.

    Args:
        image: Input image.

    Returns:
        Binary mask with values 0 or 255.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.where(gray > 0, 255, 0).astype(np.uint8)


def find_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> np.ndarray:
    """
    Find the overlapping region between two image masks.

    Args:
        mask1: Binary mask for first image.
        mask2: Binary mask for second image.

    Returns:
        Binary overlap mask.
    """
    if mask1.shape != mask2.shape:
        raise ValueError(
            "Image masks must have the same dimensions."
        )

    return cv2.bitwise_and(mask1, mask2)


def alpha_blend(
    image1: np.ndarray,
    image2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray
) -> np.ndarray:
    """
    Blend two images using distance-based weights.

    Pixels closer to the center of each image receive
    higher blending weights.

    Args:
        image1: First warped image.
        image2: Second warped image.
        mask1: Valid-pixel mask for first image.
        mask2: Valid-pixel mask for second image.

    Returns:
        Blended panorama.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            "Images must have the same dimensions."
        )

    if mask1.shape != mask2.shape:
        raise ValueError(
            "Masks must have the same dimensions."
        )

    mask1_binary = (mask1 > 0).astype(np.uint8)
    mask2_binary = (mask2 > 0).astype(np.uint8)

    distance1 = cv2.distanceTransform(
        mask1_binary,
        cv2.DIST_L2,
        5
    )

    distance2 = cv2.distanceTransform(
        mask2_binary,
        cv2.DIST_L2,
        5
    )

    total_distance = distance1 + distance2

    total_distance[
        total_distance == 0
    ] = 1.0

    weight1 = distance1 / total_distance
    weight2 = distance2 / total_distance

    weight1 = weight1[..., np.newaxis]
    weight2 = weight2[..., np.newaxis]

    image1_float = image1.astype(np.float32)
    image2_float = image2.astype(np.float32)

    blended = (
        image1_float * weight1
        + image2_float * weight2
    )

    return np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)


def simple_blend(
    image1: np.ndarray,
    image2: np.ndarray
) -> np.ndarray:
    """
    Blend two images using their valid-pixel masks.

    Args:
        image1: First warped image.
        image2: Second warped image.

    Returns:
        Combined panorama.
    """
    mask1 = create_mask(image1)
    mask2 = create_mask(image2)

    overlap = find_overlap(
        mask1,
        mask2
    )

    if np.any(overlap > 0):
        result = alpha_blend(
            image1,
            image2,
            mask1,
            mask2
        )
    else:
        result = image1.copy()

        only_image2 = (
            (mask2 > 0)
            & (mask1 == 0)
        )

        result[only_image2] = image2[only_image2]

    return result


def crop_black_borders(
    image: np.ndarray
) -> np.ndarray:
    """
    Remove unnecessary black borders around the panorama.

    Args:
        image: Input panorama.

    Returns:
        Cropped panorama.
    """
    mask = create_mask(image)

    coordinates = cv2.findNonZero(mask)

    if coordinates is None:
        return image

    x, y, width, height = cv2.boundingRect(
        coordinates
    )

    return image[
        y:y + height,
        x:x + width
    ]


def blend_panorama(
    image1: np.ndarray,
    image2: np.ndarray
) -> np.ndarray:
    """
    Complete panorama blending pipeline.

    Args:
        image1: First warped image.
        image2: Second warped image.

    Returns:
        Final cropped panorama.
    """
    blended = simple_blend(
        image1,
        image2
    )

    return crop_black_borders(blended)