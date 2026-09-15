"""
Homography estimation module.

This module estimates a projective transformation between
two sets of corresponding image points using the
Direct Linear Transform (DLT) algorithm.
"""

import numpy as np


def normalize_points(
    points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points for numerically stable DLT estimation.

    The normalization translates the centroid to the origin
    and scales the average distance from the origin to sqrt(2).

    Args:
        points: Array of points with shape (N, 2).

    Returns:
        A tuple containing:
            - normalized points
            - normalization transformation matrix
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2).")

    if len(points) < 4:
        raise ValueError(
            "At least four points are required."
        )

    centroid = np.mean(points, axis=0)

    shifted = points - centroid

    distances = np.sqrt(
        np.sum(shifted ** 2, axis=1)
    )

    mean_distance = np.mean(distances)

    if mean_distance < 1e-12:
        raise ValueError(
            "Points are degenerate and cannot be normalized."
        )

    scale = np.sqrt(2.0) / mean_distance

    transformation = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ])

    homogeneous_points = np.column_stack([
        points,
        np.ones(len(points))
    ])

    normalized_homogeneous = (
        transformation @ homogeneous_points.T
    ).T

    normalized_points = normalized_homogeneous[:, :2]

    return normalized_points, transformation


def build_dlt_matrix(
    source_points: np.ndarray,
    destination_points: np.ndarray
) -> np.ndarray:
    """
    Construct the DLT matrix used to estimate homography.

    Args:
        source_points: Corresponding points from source image.
        destination_points: Corresponding points from destination image.

    Returns:
        DLT matrix A.
    """
    if len(source_points) != len(destination_points):
        raise ValueError(
            "Source and destination points must have "
            "the same number of points."
        )

    if len(source_points) < 4:
        raise ValueError(
            "At least four point correspondences are required."
        )

    matrix_rows = []

    for (x, y), (u, v) in zip(
        source_points,
        destination_points
    ):
        matrix_rows.append([
            -x, -y, -1,
            0, 0, 0,
            u * x, u * y, u
        ])

        matrix_rows.append([
            0, 0, 0,
            -x, -y, -1,
            v * x, v * y, v
        ])

    return np.asarray(matrix_rows, dtype=np.float64)


def dlt_homography(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Estimate homography using the Direct Linear Transform algorithm.

    Args:
        source_points: Points in the source image.
        destination_points: Corresponding points in the destination image.
        normalize: Whether to normalize points before DLT.

    Returns:
        3x3 homography matrix.

    Raises:
        ValueError: If insufficient or degenerate points are provided.
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
            "Source and destination point arrays "
            "must have identical shapes."
        )

    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError(
            "Points must have shape (N, 2)."
        )

    if len(source_points) < 4:
        raise ValueError(
            "At least four correspondences are required."
        )

    if normalize:
        source_normalized, source_transform = normalize_points(
            source_points
        )

        destination_normalized, destination_transform = normalize_points(
            destination_points
        )

        matrix = build_dlt_matrix(
            source_normalized,
            destination_normalized
        )

        _, _, vh = np.linalg.svd(matrix)

        homography_normalized = vh[-1].reshape(3, 3)

        homography = (
            np.linalg.inv(destination_transform)
            @ homography_normalized
            @ source_transform
        )

    else:
        matrix = build_dlt_matrix(
            source_points,
            destination_points
        )

        _, _, vh = np.linalg.svd(matrix)

        homography = vh[-1].reshape(3, 3)

    if abs(homography[2, 2]) < 1e-12:
        raise ValueError(
            "Unable to normalize the homography matrix."
        )

    homography /= homography[2, 2]

    return homography


def project_points(
    points: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    Project 2D points using a homography matrix.

    Args:
        points: Points with shape (N, 2).
        homography: 3x3 homography matrix.

    Returns:
        Transformed points with shape (N, 2).
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "Points must have shape (N, 2)."
        )

    if homography.shape != (3, 3):
        raise ValueError(
            "Homography must have shape (3, 3)."
        )

    homogeneous_points = np.column_stack([
        points,
        np.ones(len(points))
    ])

    projected = (
        homography @ homogeneous_points.T
    ).T

    denominator = projected[:, 2]

    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError(
            "Some projected points have invalid coordinates."
        )

    projected /= denominator[:, np.newaxis]

    return projected[:, :2]


def reprojection_error(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    Calculate reprojection error for point correspondences.

    Args:
        source_points: Source image points.
        destination_points: Expected destination points.
        homography: Estimated homography.

    Returns:
        Euclidean reprojection error for each correspondence.
    """
    projected_points = project_points(
        source_points,
        homography
    )

    errors = np.linalg.norm(
        projected_points - destination_points,
        axis=1
    )

    return errors