"""
RANSAC module.

This module estimates a robust homography by repeatedly selecting
random point correspondences, computing a homography using DLT,
and identifying inliers based on reprojection error.
"""

import numpy as np

from .homography import dlt_homography, reprojection_error


def check_degenerate_points(points: np.ndarray) -> bool:
    """
    Check whether a set of points is suitable for homography estimation.

    Args:
        points: Array of 2D points.

    Returns:
        True if the points are non-degenerate, otherwise False.
    """
    if len(points) < 4:
        return True

    centered = points - np.mean(points, axis=0)

    return np.linalg.matrix_rank(centered) < 2


def ransac_homography(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    threshold: float = 4.0,
    iterations: int = 2000,
    confidence: float = 0.99,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate a robust homography using RANSAC.

    Args:
        source_points: Corresponding points from source image.
        destination_points: Corresponding points from destination image.
        threshold: Maximum reprojection error for an inlier.
        iterations: Maximum number of RANSAC iterations.
        confidence: Desired probability of finding a good model.
        random_seed: Seed for reproducible random sampling.

    Returns:
        A tuple containing:
            - best homography matrix
            - Boolean inlier mask
            - reprojection errors

    Raises:
        ValueError: If there are insufficient point correspondences.
    """
    source_points = np.asarray(
        source_points,
        dtype=np.float64
    )

    destination_points = np.asarray(
        destination_points,
        dtype=np.float64
    )

    if source_points.shape != destination_points.shape:
        raise ValueError(
            "Source and destination points must have "
            "the same shape."
        )

    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError(
            "Point arrays must have shape (N, 2)."
        )

    number_of_points = len(source_points)

    if number_of_points < 4:
        raise ValueError(
            "At least four correspondences are required "
            "for RANSAC homography estimation."
        )

    if threshold <= 0:
        raise ValueError(
            "RANSAC threshold must be greater than zero."
        )

    if iterations <= 0:
        raise ValueError(
            "Number of iterations must be greater than zero."
        )

    rng = np.random.default_rng(random_seed)

    best_homography = None
    best_inlier_mask = None
    best_inlier_count = 0
    best_error = np.inf

    required_iterations = float(iterations)

    for iteration in range(iterations):
        sample_indices = rng.choice(
            number_of_points,
            size=4,
            replace=False
        )

        sample_source = source_points[sample_indices]
        sample_destination = destination_points[sample_indices]

        if check_degenerate_points(
            sample_source
        ) or check_degenerate_points(
            sample_destination
        ):
            continue

        try:
            candidate_homography = dlt_homography(
                sample_source,
                sample_destination,
                normalize=True
            )

            errors = reprojection_error(
                source_points,
                destination_points,
                candidate_homography
            )

        except (ValueError, np.linalg.LinAlgError):
            continue

        inlier_mask = errors <= threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count == 0:
            current_error = np.inf
        else:
            current_error = float(
                np.mean(errors[inlier_mask])
            )

        is_better = (
            inlier_count > best_inlier_count
            or (
                inlier_count == best_inlier_count
                and current_error < best_error
            )
        )

        if is_better:
            best_homography = candidate_homography
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count
            best_error = current_error

            inlier_ratio = (
                best_inlier_count / number_of_points
            )

            if inlier_ratio >= 1.0:
                required_iterations = 1
            elif inlier_ratio > 0:
                probability_all_inliers = (
                    inlier_ratio ** 4
                )

                if probability_all_inliers > 0:
                    required_iterations = min(
                        required_iterations,
                        int(
                            np.ceil(
                                np.log(1 - confidence)
                                / np.log(
                                    1 - probability_all_inliers
                                )
                            )
                        )
                    )

        if iteration + 1 >= required_iterations:
            break

    if best_homography is None:
        raise ValueError(
            "RANSAC could not find a valid homography."
        )

    if best_inlier_count < 4:
        raise ValueError(
            "RANSAC found fewer than four inliers. "
            "The images may not contain enough overlap."
        )

    # Re-estimate the homography using all detected inliers.
    refined_homography = dlt_homography(
        source_points[best_inlier_mask],
        destination_points[best_inlier_mask],
        normalize=True
    )

    refined_errors = reprojection_error(
        source_points,
        destination_points,
        refined_homography
    )

    refined_inlier_mask = (
        refined_errors <= threshold
    )

    return (
        refined_homography,
        refined_inlier_mask,
        refined_errors
    )


def calculate_inlier_ratio(
    inlier_mask: np.ndarray
) -> float:
    """
    Calculate the percentage of matches classified as inliers.

    Args:
        inlier_mask: Boolean array indicating inliers.

    Returns:
        Inlier ratio between 0 and 1.
    """
    inlier_mask = np.asarray(inlier_mask)

    if inlier_mask.size == 0:
        return 0.0

    return float(
        np.mean(inlier_mask.astype(float))
    )