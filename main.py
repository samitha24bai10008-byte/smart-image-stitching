"""
Smart Image Stitching and Panorama Generation.

Command-line entry point for the Computer Vision project.
"""

import argparse
import sys
from pathlib import Path

import cv2

from src.preprocessing import preprocess_image
from src.feature_detection import detect_features
from src.feature_matching import (
    match_features,
    get_matching_points,
    draw_matches,
)
from src.ransac import (
    ransac_homography,
    calculate_inlier_ratio,
)
from src.warping import create_warped_pair
from src.blending import blend_panorama
from src.evaluation import (
    calculate_inlier_statistics,
    calculate_image_statistics,
    format_evaluation_report,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Smart Image Stitching using "
            "SIFT, DLT, RANSAC and Homography"
        )
    )

    parser.add_argument(
        "--input1",
        required=True,
        help="Path to the first input image."
    )

    parser.add_argument(
        "--input2",
        required=True,
        help="Path to the second input image."
    )

    parser.add_argument(
        "--output",
        default="output/panorama.jpg",
        help="Path for the generated panorama."
    )

    parser.add_argument(
        "--matches",
        default="output/feature_matches.jpg",
        help="Path for the feature matching visualization."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="RANSAC reprojection error threshold."
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio test threshold."
    )

    return parser.parse_args()


def ensure_parent_directory(file_path: str):
    """Create the output directory if it does not exist."""
    parent = Path(file_path).parent

    if str(parent) not in ("", "."):
        parent.mkdir(
            parents=True,
            exist_ok=True
        )


def run_pipeline(args):
    """Execute the complete image stitching pipeline."""

    print("\n==============================================")
    print("   SMART IMAGE STITCHING PROJECT")
    print("==============================================\n")

    # --------------------------------------------------
    # 1. PREPROCESSING
    # --------------------------------------------------
    print("[1/7] Loading and preprocessing images...")

    image1, gray1 = preprocess_image(args.input1)
    image2, gray2 = preprocess_image(args.input2)

    print(
        f"      Image 1: "
        f"{image1.shape[1]} x {image1.shape[0]}"
    )

    print(
        f"      Image 2: "
        f"{image2.shape[1]} x {image2.shape[0]}"
    )

    # --------------------------------------------------
    # 2. FEATURE DETECTION
    # --------------------------------------------------
    print("\n[2/7] Detecting SIFT features...")

    keypoints1, descriptors1 = detect_features(gray1)
    keypoints2, descriptors2 = detect_features(gray2)

    print(
        f"      Image 1 keypoints: "
        f"{len(keypoints1)}"
    )

    print(
        f"      Image 2 keypoints: "
        f"{len(keypoints2)}"
    )

    # --------------------------------------------------
    # 3. FEATURE MATCHING
    # --------------------------------------------------
    print("\n[3/7] Matching feature descriptors...")

    matches = match_features(
        descriptors1,
        descriptors2,
        ratio_threshold=args.ratio
    )

    print(
        f"      Reliable matches: "
        f"{len(matches)}"
    )

    # Save feature matching visualization.
    ensure_parent_directory(args.matches)

    match_visualization = draw_matches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches
    )

    cv2.imwrite(
        args.matches,
        match_visualization
    )

    print(
        f"      Match visualization saved to: "
        f"{args.matches}"
    )

    # --------------------------------------------------
    # 4. HOMOGRAPHY + RANSAC
    # --------------------------------------------------
    print("\n[4/7] Estimating homography using RANSAC + DLT...")

    points1, points2 = get_matching_points(
        keypoints1,
        keypoints2,
        matches
    )

    homography, inlier_mask, errors = ransac_homography(
        points1,
        points2,
        threshold=args.threshold
    )

    inlier_ratio = calculate_inlier_ratio(
        inlier_mask
    )

    print(
        f"      RANSAC inliers: "
        f"{int(inlier_mask.sum())}"
    )

    print(
        f"      Inlier ratio: "
        f"{inlier_ratio * 100:.2f}%"
    )

    print("\n      Estimated Homography Matrix:")
    print(homography)

    # --------------------------------------------------
    # 5. IMAGE WARPING
    # --------------------------------------------------
    print("\n[5/7] Warping images onto panorama canvas...")

    warped1, warped2 = create_warped_pair(
        image1,
        image2,
        homography
    )

    print(
        f"      Canvas size: "
        f"{warped1.shape[1]} x {warped1.shape[0]}"
    )

    # --------------------------------------------------
    # 6. BLENDING
    # --------------------------------------------------
    print("\n[6/7] Blending overlapping regions...")

    panorama = blend_panorama(
        warped1,
        warped2
    )

    # --------------------------------------------------
    # 7. SAVE + EVALUATE
    # --------------------------------------------------
    print("\n[7/7] Saving and evaluating panorama...")

    ensure_parent_directory(args.output)

    success = cv2.imwrite(
        args.output,
        panorama
    )

    if not success:
        raise RuntimeError(
            "Failed to save the panorama."
        )

    match_statistics = calculate_inlier_statistics(
        inlier_mask,
        errors
    )

    panorama_statistics = calculate_image_statistics(
        panorama
    )

    print(
        format_evaluation_report(
            match_statistics,
            panorama_statistics
        )
    )

    print(
        f"Panorama saved successfully to:\n"
        f"{args.output}"
    )

    print(
        f"Feature matches saved to:\n"
        f"{args.matches}"
    )

    return panorama


def main():
    """Main application entry point."""
    args = parse_arguments()

    try:
        run_pipeline(args)

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    except ValueError as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    except Exception as error:
        print(
            f"\nUNEXPECTED ERROR: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()