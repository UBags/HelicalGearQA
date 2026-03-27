"""
arc_detection/edge_extractor.py
================================
Extract the thin arc-edge skeleton from the gear tooth image.

Two independent edge-extraction paths are computed and then merged:

Path 1 (strong equalisation)
    CLAHE clip=20, tile=(40,40) → 4-cluster K-Means → keep darkest cluster
    → grayscale → morphological OPEN → ``morphology.thin`` → component filter

Path 2 (moderate equalisation)
    CLAHE clip=8, tile=(20,20) → 4-cluster K-Means → keep darkest cluster
    → grayscale → morphological OPEN → ``morphology.skeletonize`` → component filter

Both paths suppress components that overlap with the dilated impression
mask (``maskForRemovingEdgesCloseToImpressions``), keeping only
wide-enough strokes (width > image_width / 12).

The two results are merged with bitwise-OR, dilated slightly, then
thinned once more to produce a single clean skeleton image.

Public function
---------------
extract_arc_edges(maskedImageOriginal, overarchingMask,
                  maskForRemovingEdgesCloseToImpressions,
                  finalMasks, finalMask) → np.ndarray
"""

from __future__ import annotations

import copy

import numpy as np
import cv2
from skimage import morphology
from skimage.util import img_as_float

from preprocessing.colour_enhancement import get_rgbcolor_equalized_image
from preprocessing.segmentation_preprocess import get_kmeans_segmented_image
from utils.morphology_kernels import rectangularKernel33, rectangularKernel55


# ---------------------------------------------------------------------------
# Internal: single-path extractor
# ---------------------------------------------------------------------------

def _extract_single_path(masked_image_original: np.ndarray,
                          overarching_mask: np.ndarray,
                          mask_for_impressions: np.ndarray,
                          clip_limit: float,
                          tile_grid: tuple,
                          use_skeletonize: bool) -> np.ndarray:
    """
    Run one edge-extraction path (see module docstring for parameters).

    Parameters
    ----------
    masked_image_original : np.ndarray
        Original (pre-enhancement) masked RGB image (H, W, 3), uint8.
    overarching_mask : np.ndarray
        Boolean array (H, W) — True inside the gear-tooth region.
    mask_for_impressions : np.ndarray
        Dilated impression mask (H, W), uint8.  Components overlapping
        this mask are discarded.
    clip_limit : float
        CLAHE clip limit for the equalisation step.
    tile_grid : tuple of (int, int)
        CLAHE tile grid size.
    use_skeletonize : bool
        If True, use ``morphology.skeletonize``; otherwise use
        ``morphology.thin``.

    Returns
    -------
    np.ndarray
        Binary edge image (H, W), uint8, values 0 or 255.
    """
    img = get_rgbcolor_equalized_image(
        masked_image_original, clipLimit=clip_limit, tileGridSize=tile_grid
    )
    # Suppress pixels outside the gear-tooth mask
    img[np.logical_not(overarching_mask)] = (255, 255, 255)

    # K-Means colour quantisation — keep only the darkest cluster
    img, centers = get_kmeans_segmented_image(
        img, no_of_segments=4, resize=True, mask_cluster=None
    )

    darkest = np.array([255, 255, 255])
    for colour in centers:
        if int(np.sum(colour)) < int(np.sum(darkest)):
            darkest = colour
    darkest_tuple = (int(darkest[0]), int(darkest[1]), int(darkest[2]))

    img[np.all(img != darkest_tuple, axis=-1)] = (255, 255, 255)
    img[np.all(img == darkest_tuple, axis=-1)] = (0, 0, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Morphological open to remove noise
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, rectangularKernel55)

    # Skeletonise / thin
    sk = img_as_float(img)
    if use_skeletonize:
        sk_binary = sk < 0.5
        thinned = morphology.skeletonize(sk_binary)
    else:
        sk_binary = sk < 0.5
        thinned = morphology.thin(sk_binary)

    img = np.uint8(thinned * 255)

    # Component filter: discard narrow and impression-overlapping components
    edge_mask = np.zeros(img.shape, dtype=np.uint8)
    _, width = img.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, 4)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        comp = (labels == i).astype(np.uint8) * 255
        overlaps_impression = np.sum(
            np.logical_and(mask_for_impressions, comp)
        ) != 0
        if (w > width / 12) and not overlaps_impression:
            edge_mask = cv2.bitwise_or(edge_mask, comp)

    return edge_mask


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_arc_edges(masked_image_original: np.ndarray,
                       overarching_mask: np.ndarray,
                       mask_for_impressions: np.ndarray,
                       final_masks: list,
                       final_mask: np.ndarray) -> np.ndarray:
    """
    Extract the thin arc-edge skeleton from *masked_image_original*.

    If no impression masks were found (``final_masks`` is empty) the
    function returns *final_mask* directly as a fallback (preserving
    the original notebook behaviour).

    Parameters
    ----------
    masked_image_original : np.ndarray
        Original (pre-enhancement) masked RGB image (H, W, 3), uint8.
    overarching_mask : np.ndarray
        Boolean array (H, W) — True inside the gear-tooth region.
    mask_for_impressions : np.ndarray
        Dilated impression mask from ``impression_refiner`` (H, W), uint8.
    final_masks : list
        List of impression masks from ``impression_refiner``.
        Used only to decide whether to run the full extraction pipeline
        (non-empty) or fall back to *final_mask* (empty).
    final_mask : np.ndarray
        Combined impression mask (H, W), uint8.  Used as fallback and
        also OR'd into the final result.

    Returns
    -------
    np.ndarray
        Binary arc-edge skeleton image (H, W), uint8, values 0 or 255.
    """
    if not final_masks:
        return final_mask

    # Path 1: strong equalisation + thin
    path1 = _extract_single_path(
        masked_image_original, overarching_mask, mask_for_impressions,
        clip_limit=20.0, tile_grid=(40, 40), use_skeletonize=False
    )

    # Path 2: moderate equalisation + skeletonize
    path2 = _extract_single_path(
        masked_image_original, overarching_mask, mask_for_impressions,
        clip_limit=8.0, tile_grid=(20, 20), use_skeletonize=True
    )

    # Merge both paths
    merged = cv2.bitwise_or(path1, path2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_DILATE, rectangularKernel33)

    # Final thinning pass (sk > 0.5 because foreground is white after merge)
    sk = img_as_float(merged)
    sk_binary = sk > 0.5
    thinned = morphology.thin(sk_binary)
    arc_edges = np.uint8(thinned * 255)

    # OR in the impression mask so arc contours that pass through
    # impressions are not lost
    arc_edges = cv2.bitwise_or(arc_edges, final_mask)

    return arc_edges