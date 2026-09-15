"""
Evaluation module.

Provides quantitative measurements for evaluating the
image stitching pipeline.
"""

from typing import Optional

import cv2
import numpy as np


def calculate_inlier_statistics(
    inlier_mask: np.ndarray,
    errors: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate statistics for RANSAC inliers.

    Args:
        inlier_mask: Boolean mask indicating valid matches.
        errors: Optional reprojection errors.

    Returns:
        Dictionary containing evaluation statistics.
    """
    inlier_mask = np.asarray(inlier_mask).astype(bool)

    total_matches = len(inlier_mask)
    inlier_count = int(np.sum(inlier_mask))

    if total_matches == 0:
        return {
            "total_matches": 0,
            "inlier_count": 0,
            "outlier_count": 0,
            "inlier_ratio": 0.0,
            "mean_reprojection_error": None
        }

    inlier_ratio = inlier_count / total_matches
    outlier_count = total_matches - inlier_count

    mean_error = None

    if errors is not None and inlier_count > 0:
        errors = np.asarray(errors)
        mean_error = float(
            np.mean(errors[inlier_mask])
        )

    return {
        "total_matches": total_matches,
        "inlier_count": inlier_count,
        "outlier_count": outlier_count,
        "inlier_ratio": float(inlier_ratio),
        "mean_reprojection_error": mean_error
    }


def calculate_sharpness(
    image: np.ndarray
) -> float:
    """
    Calculate image sharpness using the variance
    of the Laplacian.

    Higher values generally indicate a sharper image.

    Args:
        image: Input image.

    Returns:
        Sharpness score.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def calculate_brightness(
    image: np.ndarray
) -> float:
    """
    Calculate the average brightness of an image.

    Args:
        image: Input image.

    Returns:
        Mean grayscale intensity.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    return float(
        np.mean(gray)
    )


def calculate_image_statistics(
    image: np.ndarray
) -> dict:
    """
    Calculate basic image quality statistics.

    Args:
        image: Input image.

    Returns:
        Dictionary containing image dimensions,
        brightness and sharpness.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image.")

    height, width = image.shape[:2]

    return {
        "width": int(width),
        "height": int(height),
        "channels": (
            int(image.shape[2])
            if len(image.shape) == 3
            else 1
        ),
        "brightness": calculate_brightness(image),
        "sharpness": calculate_sharpness(image)
    }


def compare_image_quality(
    input_image: np.ndarray,
    panorama: np.ndarray
) -> dict:
    """
    Compare basic quality measurements between an
    input image and the resulting panorama.

    Args:
        input_image: Original input image.
        panorama: Generated panorama.

    Returns:
        Dictionary containing quality measurements.
    """
    input_stats = calculate_image_statistics(
        input_image
    )

    panorama_stats = calculate_image_statistics(
        panorama
    )

    return {
        "input": input_stats,
        "panorama": panorama_stats
    }


def format_evaluation_report(
    match_statistics: dict,
    panorama_statistics: dict
) -> str:
    """
    Convert evaluation results into a readable report.

    Args:
        match_statistics: Feature matching statistics.
        panorama_statistics: Panorama statistics.

    Returns:
        Formatted text report.
    """
    mean_error = match_statistics[
        "mean_reprojection_error"
    ]

    error_text = (
        f"{mean_error:.4f} pixels"
        if mean_error is not None
        else "N/A"
    )

    return (
        "\n"
        "========== IMAGE STITCHING EVALUATION ==========\n"
        f"Total feature matches      : "
        f"{match_statistics['total_matches']}\n"
        f"RANSAC inliers             : "
        f"{match_statistics['inlier_count']}\n"
        f"RANSAC outliers            : "
        f"{match_statistics['outlier_count']}\n"
        f"Inlier ratio               : "
        f"{match_statistics['inlier_ratio'] * 100:.2f}%\n"
        f"Mean reprojection error    : "
        f"{error_text}\n"
        f"Panorama width             : "
        f"{panorama_statistics['width']} pixels\n"
        f"Panorama height            : "
        f"{panorama_statistics['height']} pixels\n"
        f"Panorama brightness        : "
        f"{panorama_statistics['brightness']:.2f}\n"
        f"Panorama sharpness         : "
        f"{panorama_statistics['sharpness']:.2f}\n"
        "=================================================\n"
    )