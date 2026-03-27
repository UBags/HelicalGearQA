"""
utils/morphology_kernels.py
============================
Pre-built OpenCV structuring elements (kernels) used throughout the
pipeline for morphological operations (erosion, dilation, opening,
closing, thinning).

Centralising them here avoids re-creating them on every function call
and gives them consistent names.

Naming convention
-----------------
  ellipticalKernel<rows><cols>   – cv2.MORPH_ELLIPSE shaped kernel
  rectangularKernel<rows><cols>  – rectangular (all-ones) kernel

Import example
--------------
    from utils.morphology_kernels import (
        ellipticalKernel33, ellipticalKernel55,
        rectangularKernel55,
    )
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Elliptical kernels
# ---------------------------------------------------------------------------
ellipticalKernel33:    np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ellipticalKernel55:    np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
ellipticalKernel77:    np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
ellipticalKernel99:    np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Wide flat kernel used to dilate the impression mask so that arc-edge
# components that overlap with impressions can be discarded.
ellipticalKernel81_15: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 15))

# ---------------------------------------------------------------------------
# Rectangular kernels  (np.uint8 all-ones arrays)
# ---------------------------------------------------------------------------
rectangularKernel33: np.ndarray = np.ones((3, 3), np.uint8)
rectangularKernel55: np.ndarray = np.ones((5, 5), np.uint8)
rectangularKernel77: np.ndarray = np.ones((7, 7), np.uint8)
rectangularKernel99: np.ndarray = np.ones((9, 9), np.uint8)

# Tall-and-narrow kernel (9 rows × 5 cols) — used for directional morphology.
rectangularKernel95: np.ndarray = np.ones((9, 5), np.uint8)