"""
Feature matching module.

Matches feature descriptors between two images using
the SIFT descriptor and Lowe's ratio test.
"""

import cv2
import numpy as np


def create_matcher() -> cv2.BFMatcher:
    """
    Create a Brute-Force matcher suitable for SIFT descriptors.

    Returns:
        Configured BFMatcher.
    """
    return cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False
    )


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.75
) -> list[cv2.DMatch]:
    """
    Match SIFT descriptors using Lowe's ratio test.

    Args:
        descriptors1: Descriptors from the first image.
        descriptors2: Descriptors from the second image.
        ratio_threshold: Threshold used to reject ambiguous matches.

    Returns:
        List of reliable feature matches.

    Raises:
        ValueError: If descriptors are invalid or too few matches
                    are found.
    """
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Feature descriptors cannot be None.")

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        raise ValueError("Not enough descriptors for matching.")

    matcher = create_matcher()

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        best_match, second_match = pair

        if best_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(best_match)

    if len(good_matches) < 4:
        raise ValueError(
            f"Only {len(good_matches)} reliable matches found. "
            "Use images with sufficient overlap."
        )

    good_matches.sort(key=lambda match: match.distance)

    return good_matches


def get_matching_points(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract corresponding point coordinates from matched keypoints.

    Args:
        keypoints1: Keypoints from the first image.
        keypoints2: Keypoints from the second image.
        matches: Reliable feature matches.

    Returns:
        Two arrays containing corresponding points.
    """
    if len(matches) < 4:
        raise ValueError(
            "At least four matches are required."
        )

    points1 = np.float32([
        keypoints1[match.queryIdx].pt
        for match in matches
    ])

    points2 = np.float32([
        keypoints2[match.trainIdx].pt
        for match in matches
    ])

    return points1, points2


def draw_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    max_matches: int = 50
) -> np.ndarray:
    """
    Create a visualization of feature matches.

    Args:
        image1: First image.
        keypoints1: Keypoints from first image.
        image2: Second image.
        keypoints2: Keypoints from second image.
        matches: Feature matches.
        max_matches: Maximum matches to display.

    Returns:
        Combined image showing feature correspondences.
    """
    selected_matches = matches[:max_matches]

    return cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        selected_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )