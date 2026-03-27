"""
utils/image_io.py
==================
Utility helpers for:
  - Loading images from disk (BMP / PNG / JPG → RGB numpy array)
  - Iterating over all image files in the configured input directory
  - Per-component configuration lookups (arc length, tolerances,
    width multiplier, prompt points) — these were scattered at module
    level in the original notebook

All path defaults come from config.settings.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import (
    IMAGES_DIRECTORY,
    theInputPoints,
    arcLengthsDict,
    arcLengthsToleranceDict,
    widthMultiplierDict,
    DEFAULT_ARC_LENGTH_MM,
    DEFAULT_TOLERANCES,
    DEFAULT_WIDTH_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load an image from *image_path* and return it as an RGB numpy array
    (dtype uint8, shape H×W×3).

    OpenCV reads in BGR by default; this function converts to RGB so that
    all downstream code works in a consistent colour space.

    Parameters
    ----------
    image_path : str
        Full path to the image file (BMP, PNG, JPG, etc.).

    Returns
    -------
    np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be read by OpenCV.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(
            f"cv2.imread could not read '{image_path}'. "
            "Check that the file exists and is a supported format."
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------

def iter_image_files(directory: str = IMAGES_DIRECTORY):
    """
    Yield ``pathlib.Path`` objects for every file in *directory*.

    Parameters
    ----------
    directory : str
        Path to the folder containing the input images.
        Defaults to ``config.settings.IMAGES_DIRECTORY``.

    Yields
    ------
    pathlib.Path
        One Path per file found (non-recursive).

    Raises
    ------
    NotADirectoryError
        If *directory* does not exist.
    """
    base = Path(directory)
    if not base.is_dir():
        raise NotADirectoryError(
            f"Image directory '{directory}' does not exist. "
            "Update IMAGES_DIRECTORY in config/settings.py."
        )
    for entry in base.iterdir():
        if entry.is_file():
            yield entry


# ---------------------------------------------------------------------------
# Per-component configuration lookups
# ---------------------------------------------------------------------------

def get_input_points(filename: str) -> Optional[list]:
    """
    Return the SAM prompt points for the component identified by *filename*.

    The lookup searches for a key from ``theInputPoints`` that is a
    *substring* of *filename*.  Returns ``None`` if no match is found,
    which tells the caller to use the automatic (no-prompt) SAM pipeline.

    Parameters
    ----------
    filename : str
        The base name (or full path) of the image file.

    Returns
    -------
    list or None
        Prompt-point list in SAM format, e.g.
        ``[ [[x1, y1], [x2, y2], [x3, y3]] ]``.
    """
    for key, points in theInputPoints.items():
        if key in filename:
            return points
    return None


def get_arc_length_mm(filename: str) -> float:
    """
    Return the known physical arc length (mm) for the component whose
    identifier appears as a substring of *filename*.

    Falls back to ``DEFAULT_ARC_LENGTH_MM`` (64.0 mm) if not found.

    Parameters
    ----------
    filename : str
        Image file name used for component identification.

    Returns
    -------
    float
        Arc length in millimetres.
    """
    for key, length in arcLengthsDict.items():
        if key in filename:
            return length
    return DEFAULT_ARC_LENGTH_MM


def get_arc_length_tolerances(filename: str) -> list:
    """
    Return the measurement tolerance table for the component identified
    by *filename*.

    Each entry is a [min, max] pair (mm) for:
        [0] Toe Clearance
        [1] Contact Length
        [2] Heel Clearance
        [3] Contact Width
        [4] Tip Clearance

    Falls back to ``DEFAULT_TOLERANCES`` if not found.

    Parameters
    ----------
    filename : str
        Image file name used for component identification.

    Returns
    -------
    list of [float, float]
        Five [min, max] pairs.
    """
    for key, tolerances in arcLengthsToleranceDict.items():
        if key in filename:
            return tolerances
    return DEFAULT_TOLERANCES


def get_width_multiplier(filename: str) -> float:
    """
    Return the width scaling multiplier for the component identified by
    *filename*.

    Falls back to ``DEFAULT_WIDTH_MULTIPLIER`` (1.35) if not found.

    Parameters
    ----------
    filename : str
        Image file name used for component identification.

    Returns
    -------
    float
        Width multiplier (dimensionless).
    """
    for key, multiplier in widthMultiplierDict.items():
        if key in filename:
            return multiplier
    return DEFAULT_WIDTH_MULTIPLIER