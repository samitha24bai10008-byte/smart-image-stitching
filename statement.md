# Project Statement

## Project Title

Smart Image Stitching and Panorama Generation using Computer Vision

## Problem Statement

Capturing a wide scene using a single photograph can be difficult when
the camera has a limited field of view. Multiple overlapping photographs
can be combined to create a wider panoramic image.

The objective of this project is to develop a computer vision system
that automatically stitches two overlapping images into a single
panoramic image.

The system identifies common visual features between the images,
estimates the geometric relationship between them, removes incorrect
feature correspondences, warps the images into a common coordinate
system, and blends them to produce the final panorama.

## Scope of the Project

The project focuses on classical computer vision techniques for
two-image panorama generation.

The system includes:

- Image preprocessing
- SIFT feature detection
- Feature descriptor matching
- Lowe's ratio test
- Homography estimation using DLT
- RANSAC-based outlier rejection
- Perspective image warping
- Image blending
- Panorama quality evaluation

The current implementation accepts two overlapping images as input.

## Target Users

The project can be useful for:

- Students learning Computer Vision
- Researchers experimenting with image stitching
- Developers interested in classical computer vision
- Users who want to combine overlapping photographs into a panorama

## High-Level Features

1. Automatic image loading and validation
2. Image preprocessing and resizing
3. SIFT keypoint and descriptor extraction
4. Feature matching between images
5. Homography estimation using DLT
6. RANSAC-based removal of incorrect matches
7. Perspective transformation and image warping
8. Image blending
9. Automatic panorama generation
10. Quantitative result evaluation
11. Command-line execution
12. Feature matching visualization