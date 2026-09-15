# Smart Image Stitching and Panorama Generation

## 1. Project Overview

This project implements an automatic image stitching system for generating
a panoramic image from two overlapping images.

The system uses classical Computer Vision techniques to detect image
features, find corresponding points, estimate a geometric transformation,
remove incorrect matches, warp the images, and blend them into a panorama.

### Main techniques used

- Image preprocessing
- SIFT feature detection
- Feature descriptor matching
- Lowe's ratio test
- Direct Linear Transform (DLT)
- Homography estimation
- RANSAC outlier rejection
- Perspective image warping
- Distance-based image blending
- Panorama evaluation

---

## 2. Project Structure

```text
smart-image-stitching/
│
├── input/
│   ├── image1.jpg
│   └── image2.jpg
│
├── output/
│   ├── panorama.jpg
│   └── feature_matches.jpg
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_detection.py
│   ├── feature_matching.py
│   ├── homography.py
│   ├── ransac.py
│   ├── warping.py
│   ├── blending.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
└── README.md