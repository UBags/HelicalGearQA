"""
segmentation/impression_detector.py
=====================================
Detect the gear-tooth contact impressions (the black ink marks left on
the gear teeth during the blue-dye test).

The detection pipeline operates entirely on the masked/equalized image
produced by ``segmentation.overarching_mask`` and produces a
``preFinalMask`` and ``finalCentroids`` list that feed into the SAM
Pass 2 refinement step (``segmentation.impression_refiner``).

Public functions
----------------
get_otsu_threshold
    Custom Otsu thresholding on a 1-D pixel array using a histogram.

detect_impressions
    Full impression detection pipeline.  Returns ``preFinalMask``,
    ``finalCentroids``, ``areaLimits``, and the display image
    (``enhancedYellowsAndBlacksImage``).
"""

from __future__ import annotations

import copy

import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.segmentation import chan_vese
from skimage.util import img_as_float, img_as_ubyte

from preprocessing.colour_enhancement import enhance_yellows_and_blacks
from utils.morphology_kernels import (
    ellipticalKernel33,
    ellipticalKernel55,
    ellipticalKernel77,
)


# ---------------------------------------------------------------------------
# Otsu threshold
# ---------------------------------------------------------------------------

def get_otsu_threshold(a_1d_array: np.ndarray,
                        number_of_bins: int = 52,
                        adjustment: int = 12) -> int:
    """
    Compute an Otsu-style threshold from *a_1d_array* using a histogram.

    This is a custom implementation that matches the original notebook's
    ``getOtsuThreshold`` function.  It uses *number_of_bins* histogram
    bins rather than 256, which stabilises the threshold on small or
    noisy arrays.

    Parameters
    ----------
    a_1d_array : np.ndarray
        1-D (or flat) array of pixel intensities.
    number_of_bins : int
        Number of histogram bins (default 52).
    adjustment : int
        Constant subtracted from the raw Otsu threshold to bias it
        towards darker regions (default 12).

    Returns
    -------
    int
        Threshold value, clamped to [0, 254].
    """
    counts, bin_edges = np.histogram(a_1d_array, bins=number_of_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1   = np.cumsum(counts * bin_centers) / weight1
    mean2   = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx        = int(np.argmax(variance12))
    threshold  = bin_centers[idx]

    return min(max(0, int(threshold) - adjustment), 254)


# ---------------------------------------------------------------------------
# Main impression detection pipeline
# ---------------------------------------------------------------------------

def detect_impressions(masked_image: np.ndarray,
                        masked_image_equalized: np.ndarray,
                        overarching_mask: np.ndarray,
                        overarching_mask_width: int,
                        overarching_mask_height: int) -> dict:
    """
    Detect contact impressions within the gear-tooth region.

    Pipeline steps
    --------------
    1. **Chan-Vese segmentation** on the equalized/blurred image to split
       the tooth into background and contact zones.  Large Chan-Vese
       components are retained.
    2. **Yellow/Black enhancement** applied to the masked image after
       suppressing the Chan-Vese background.  This isolates the
       yellow-and-black impression pattern.
    3. **Black-ratio filtering** — each connected component is scored by
       the fraction of dark pixels.  Components with 15–58 % dark pixels
       are kept (they are mixed yellow/black = the impression zone).
    4. **White-area component filtering** — a second pass on the
       white-to-gray inverted image filters by size constraints.
    5. **Morphological cleanup** — open, close, erode, dilate sequence to
       smooth the retained mask.
    6. **Final component selection** — keep components within strict size
       and area bounds, record their centroids and area limits for SAM
       Pass 2.

    Parameters
    ----------
    masked_image : np.ndarray
        Gamma/contrast/colour-enhanced masked image (RGB, uint8).
        From the enhancement chain in ``overarching_mask.compute_overarching_mask``.
    masked_image_equalized : np.ndarray
        Heavily blurred and equalised version of the masked image.
        From ``overarching_mask.compute_overarching_mask``.
    overarching_mask : np.ndarray
        Boolean array (H, W) — True inside the gear-tooth region.
    overarching_mask_width : int
        Width of the gear-tooth bounding box in pixels.
    overarching_mask_height : int
        Height of the gear-tooth bounding box in pixels.

    Returns
    -------
    dict with keys:
        ``preFinalMask`` : np.ndarray
            uint8 grayscale image (H, W) — white where candidate
            impressions were found.
        ``finalCentroids`` : list
            Centroids in SAM prompt format:
            ``[ [ [[cx1, cy1]], [[cx2, cy2]], … ] ]``
        ``areaLimits`` : list of [float, float]
            ±5 % area bands for each centroid, used to validate SAM masks.
        ``enhancedYellowsAndBlacksImage`` : np.ndarray
            The yellow/black enhanced image for display (axes[0]).
    """
    # ------------------------------------------------------------------ #
    # Step 1: Chan-Vese segmentation
    # ------------------------------------------------------------------ #
    img_eq = copy.deepcopy(masked_image_equalized)

    gray_float = rgb2gray(img_as_float(img_eq))
    cv_result  = chan_vese(gray_float, max_num_iter=50, extended_output=True)

    # cv_result[1] is the signed distance function (SDF) of the level set
    cv_seg = cv_result[1]
    cv_seg = np.clip(cv_seg, -1.0, 1.0)
    cv_seg = img_as_ubyte(cv_seg)   # maps [-1,1] → [0, 255]
    cv_seg = 255 - cv_seg

    thresh_cv = get_otsu_threshold(cv_seg, adjustment=0)
    thresh_cv = int(thresh_cv * 0.65)
    cv_seg[cv_seg <= thresh_cv] = 0
    cv_seg[cv_seg > thresh_cv]  = 255

    cv_seg = 255 - cv_seg   # invert: impressions are now white

    # Keep only Chan-Vese components large enough to be impressions
    retained_cv = np.zeros(cv_seg.shape, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cv_seg, 4)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (w > overarching_mask_width / 3) and (h > overarching_mask_height / 10):
            comp = (labels == i).astype(np.uint8) * 255
            retained_cv = cv2.bitwise_or(retained_cv, comp)

    cv_bool = (( 255 - retained_cv) / 255).astype(bool)

    # ------------------------------------------------------------------ #
    # Step 2: Yellow/Black enhancement on impression candidates
    # ------------------------------------------------------------------ #
    img_work = copy.deepcopy(masked_image)
    img_work[cv_bool] = (255, 255, 255)           # suppress Chan-Vese background

    img_work = enhance_yellows_and_blacks(
        img_work,
        kernelSize=5, percentileCutoff=20,
        changeBlacks=False, changeYellows=True,
        rgCutoff=(35, 38), rgbDiff=9
    )
    enhanced_yb_image = copy.deepcopy(img_work)   # saved for display

    # ------------------------------------------------------------------ #
    # Step 3: Black-ratio filtering
    # ------------------------------------------------------------------ #
    yb_gray = cv2.cvtColor(enhanced_yb_image, cv2.COLOR_RGB2GRAY)
    yb_gray_inv = 255 - yb_gray
    yb_gray_inv[yb_gray_inv > 10] = 255

    retained_br = np.zeros(yb_gray_inv.shape, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(yb_gray_inv, 4)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        comp_mask  = (labels == i).astype(np.uint8) * 255
        comp_bool  = (comp_mask / 255).astype(bool)
        patch      = copy.deepcopy(img_work)
        patch[~comp_bool] = (255, 255, 255)

        r_p, g_p, b_p = cv2.split(patch)
        dark_pixels   = np.logical_and(r_p < 175, g_p < 175, b_p < 175)   # type: ignore[arg-type]
        blacks        = int(np.sum(dark_pixels))
        black_pct     = blacks * 100.0 / area if area > 0 else 0

        keep = (black_pct < 58) and (black_pct >= 15)
        if keep:
            retained_br = cv2.bitwise_or(retained_br, comp_mask)
        elif area > 2000:
            print(f"[impression_detector] Discarded area={area} (black%={black_pct:.1f})")

    retained_br_bool = (retained_br / 255).astype(bool)
    img_work[~retained_br_bool] = (255, 255, 255)

    # ------------------------------------------------------------------ #
    # Step 4: White-area component size filter
    # ------------------------------------------------------------------ #
    r_w, g_w, b_w = cv2.split(img_work)
    white_mask = np.logical_and(r_w > 175, g_w > 175, b_w > 175)
    img_work[white_mask] = (255, 255, 255)

    gray_work = cv2.cvtColor(img_work, cv2.COLOR_RGB2GRAY)
    white_segs = 255 - gray_work
    white_segs[white_segs > 10] = 255

    retained_sz = np.zeros(white_segs.shape, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_segs, 4)

    for i in range(1, num_labels):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keep = (
            (w    > overarching_mask_width  / 4) and
            (h    > overarching_mask_height / 20) and
            (area > 2000) and
            (area < 6000) and
            (w    < overarching_mask_width  * 0.85)
        )
        if keep:
            comp = (labels == i).astype(np.uint8) * 255
            retained_sz = cv2.bitwise_or(retained_sz, comp)
        elif area > 2000:
            print(f"[impression_detector] Size-filter discarded area={area}")

    # ------------------------------------------------------------------ #
    # Step 5: Morphological cleanup
    # ------------------------------------------------------------------ #
    retained_sz = cv2.morphologyEx(retained_sz, cv2.MORPH_OPEN,   ellipticalKernel33)
    retained_sz = cv2.morphologyEx(retained_sz, cv2.MORPH_CLOSE,  ellipticalKernel33)
    retained_sz = cv2.morphologyEx(retained_sz, cv2.MORPH_ERODE,  ellipticalKernel55)
    retained_sz = cv2.morphologyEx(retained_sz, cv2.MORPH_DILATE, ellipticalKernel77)

    # ------------------------------------------------------------------ #
    # Step 6: Final component selection → preFinalMask + centralCentroids
    # ------------------------------------------------------------------ #
    pre_final_mask  = np.zeros(retained_sz.shape, dtype=np.uint8)
    final_centroids: list = []
    area_limits:     list = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(retained_sz, 4)

    for i in range(1, num_labels):
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keep = (
            (w    > overarching_mask_width  / 4) and
            (h    > overarching_mask_height / 20) and
            (area > 2000) and
            (area < 6500)
        )
        if keep:
            print(f"[impression_detector] Component retained: area={area}")
            comp           = (labels == i).astype(np.uint8) * 255
            pre_final_mask = cv2.bitwise_or(pre_final_mask, comp)
            final_centroids.append([list(centroids[i])])
            area_limits.append([area * 0.95, area * 1.05])

    return {
        "preFinalMask":                    pre_final_mask,
        "finalCentroids":                  final_centroids,
        "areaLimits":                      area_limits,
        "enhancedYellowsAndBlacksImage":   enhanced_yb_image,
    }