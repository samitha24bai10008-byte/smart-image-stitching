"""
Feature detection module.

Detects distinctive keypoints and computes feature descriptors
using the SIFT algorithm.
"""

import cv2
import numpy as np


def create_sift_detector(
    n_features: int = 2000
) -> cv2.SIFT:
    """
    Create a SIFT feature detector.

    Args:
        n_features: Maximum number of features to retain.

    Returns:
        Configured SIFT detector.
    """
    return cv2.SIFT_create(nfeatures=n_features)


def detect_features(
    gray_image: np.ndarray,
    n_features: int = 2000
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """
    Detect SIFT keypoints and compute their descriptors.

    Args:
        gray_image: Grayscale input image.
        n_features: Maximum number of features.

    Returns:
        A tuple containing:
            - detected keypoints
            - feature descriptors

    Raises:
        ValueError: If the input image is invalid or
                    insufficient features are detected.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Invalid grayscale image.")

    detector = create_sift_detector(n_features)

    keypoints, descriptors = detector.detectAndCompute(
        gray_image,
        None
    )

    if keypoints is None or len(keypoints) < 4:
        raise ValueError(
            "Insufficient features detected. "
            "Use an image with more texture or detail."
        )

    if descriptors is None:
        raise ValueError("Feature descriptors could not be computed.")

    return keypoints, descriptors


def draw_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint]
) -> np.ndarray:
    """
    Draw detected keypoints on an image.

    Args:
        image: Input BGR image.
        keypoints: Detected SIFT keypoints.

    Returns:
        Image containing visualized keypoints.
    """
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def keypoints_to_array(
    keypoints: list[cv2.KeyPoint]
) -> np.ndarray:
    """
    Convert OpenCV keypoints into an array of coordinates.

    Args:
        keypoints: List of OpenCV keypoints.

    Returns:
        NumPy array of shape (N, 2).
    """
    return np.float32([
        keypoint.pt for keypoint in keypoints
    ])