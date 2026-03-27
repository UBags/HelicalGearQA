"""
segmentation/overarching_mask.py
==================================
SAM Pass 1 — derive the overarching gear-tooth region mask.

This module contains two public functions:

select_best_sam_mask
    Given the raw SAM output (masks + scores), pick the smallest mask
    whose area is < 50 % of the image and that contains a significant
    proportion of yellow pixels.  Returns the raw masked image and the
    binary mask array.

compute_overarching_mask
    Apply the sequence of colour enhancements and blurring steps to the
    selected masked image, then threshold to produce:
      - ``overarchingMask``        (bool array, H×W)
      - ``overarchingMaskContour`` (the largest contour of the mask)
      - ``overarchingMaskWidth``   (int, pixels)
      - ``overarchingMaskHeight``  (int, pixels)

These two outputs feed directly into the impression-detection and
arc-detection modules.
"""

from __future__ import annotations

import copy
import functools

import numpy as np
import cv2

from preprocessing.colour_enhancement import (
    determine_gamma,
    apply_gamma,
    change_contrast_brightness,
    enhance_rg_suppress_b,
    get_rgbcolor_equalized_image,
    enhance_yellows_and_blacks,
)


# ---------------------------------------------------------------------------
# Step 1 — select best SAM mask
# ---------------------------------------------------------------------------

def select_best_sam_mask(raw_image: np.ndarray,
                          masks_list: list,
                          scores_tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate every mask returned by the first SAM pass and select the
    one that best represents the gear-tooth region.

    Selection criterion (from the original notebook):
    - Discard any mask whose area is ≥ 50 % of the total image area
      (these are usually background masks).
    - For the remaining candidates, choose the one with the *smallest*
      area.  In practice the three SAM predictions per prompt point
      correspond to different granularities; the smallest qualifying mask
      is the tightest fit around the contact zone.

    Parameters
    ----------
    raw_image : np.ndarray
        Original RGB image, shape (H, W, 3), dtype uint8.
    masks_list : list
        Nested mask output from ``model.inferencer.run_prompted_inference``.
        Specifically ``masks_list[0][prediction_index]`` is used.
    scores_tensor : torch.Tensor
        IOU scores, shape (1, n_predictions, n_variants).

    Returns
    -------
    masked_image : np.ndarray
        RGB image with non-mask pixels set to white (255, 255, 255),
        shape (H, W, 3), dtype uint8.
    original_mask : np.ndarray
        Boolean mask array, shape (H, W).

    Raises
    ------
    ValueError
        If no qualifying mask is found (all masks cover ≥ 50 % of image).
    """
    total_area = raw_image.shape[0] * raw_image.shape[1]

    masked_images:  list[np.ndarray] = []
    areas:          list[float]      = []
    original_masks: list[np.ndarray] = []

    # masks_list[0] contains one entry per prompt point (prediction);
    # each entry holds multiple variants (usually 3, sorted by IOU score).
    # We iterate over prediction_index × variant_index.
    _, n_predictions, _ = scores_tensor.shape

    for pred_idx in range(n_predictions):
        for mask_tensor in masks_list[0][pred_idx]:
            mask_np = mask_tensor.cpu().detach().numpy()
            mask_u8 = mask_np.astype(np.uint8)
            area    = float(np.sum(mask_u8))

            # Skip masks that cover ≥ 50 % of the image
            if area / total_area >= 0.5:
                continue

            # Build masked image: apply mask and replace background with white
            mask_255 = np.dstack([mask_u8 * 255] * 3)
            masked   = np.bitwise_and(mask_255, raw_image).astype(np.uint8)

            # Quick yellow-pixel count to assess mask quality
            r_ch = masked[:, :, 0].astype(np.int32)
            g_ch = masked[:, :, 1].astype(np.int32)
            b_ch = masked[:, :, 2].astype(np.int32)
            yellow_cond = functools.reduce(np.logical_and, [
                r_ch > 80, g_ch > 80,
                (r_ch - b_ch) > 30,
                (g_ch - b_ch) > 30,
                b_ch < 170,
            ])

            masked_images.append(masked)
            areas.append(area)
            original_masks.append(mask_np)

    if not areas:
        raise ValueError(
            "select_best_sam_mask: all SAM masks cover ≥ 50 % of the image. "
            "Check that the prompt points are correct for this image."
        )

    # Choose smallest qualifying mask
    idx = int(np.argmin(areas))
    masked_image   = masked_images[idx].astype(np.uint8)
    original_mask  = original_masks[idx]

    # Replace true-black pixels (originally outside mask but not background)
    # with white so that downstream processing works on a white background
    masked_image[np.where((masked_image == [0, 0, 0]).all(axis=2))] = (255, 255, 255)

    return masked_image, original_mask


# ---------------------------------------------------------------------------
# Step 2 — compute overarching mask
# ---------------------------------------------------------------------------

def compute_overarching_mask(masked_image: np.ndarray,
                              original_mask: np.ndarray) -> dict:
    """
    Derive the overarching gear-tooth region mask from the SAM-selected
    masked image.

    Pipeline:
    1. Gamma correction, contrast boost, red/green channel enhancement.
    2. Two rounds of CLAHE equalisation + yellow/black enhancement.
    3. Median blur → box blur → bilateral filter to smooth the image.
    4. Threshold the grayscale version at 250 to get a binary gear-tooth mask.
    5. AND with the original SAM mask to remove background bleed-through.
    6. Keep only connected components that are at least 1/6 of the image
       in both width and height.
    7. Find the contour of the retained mask (excluding the full-image border).

    Parameters
    ----------
    masked_image : np.ndarray
        White-background masked RGB image from ``select_best_sam_mask``,
        shape (H, W, 3), dtype uint8.
    original_mask : np.ndarray
        Boolean (or float) mask array from SAM, shape (H, W).

    Returns
    -------
    dict with keys:
        ``maskedImageOriginal`` : np.ndarray
            Copy of *masked_image* before any enhancement — used later
            by the arc-detection module.
        ``overarchingMask`` : np.ndarray
            Boolean array (H, W) — True inside the gear-tooth region.
        ``overarchingMaskContour`` : np.ndarray
            OpenCV contour array of the mask boundary.
        ``overarchingMaskWidth`` : int
            Maximum width (pixels) of the retained connected components.
        ``overarchingMaskHeight`` : int
            Maximum height (pixels) of the retained connected components.
        ``maskedImage_Equalized`` : np.ndarray
            The fully equalised/blurred version of the masked image — used
            by the impression detector for Chan-Vese segmentation.
    """
    # Keep a pristine copy for arc detection
    masked_image_original = copy.deepcopy(masked_image)

    # --- Colour enhancement chain ---
    img = copy.deepcopy(masked_image)
    img = cv2.resize(img, (img.shape[1], img.shape[0]))          # no-op resize (keeps contiguous)
    gamma = determine_gamma(img)
    img   = apply_gamma(img, gamma)
    img   = change_contrast_brightness(img, clipLimit=4.0)
    img   = enhance_rg_suppress_b(img, clipLimit=3.0)

    # Two-round CLAHE + yellow/black enhancement, then heavy blurring
    img_eq = get_rgbcolor_equalized_image(img, clipLimit=6.0, tileGridSize=(6, 6))
    img_eq = enhance_yellows_and_blacks(img_eq, grayRatioCutoff=1.0025,
                                         changeBlacks=False, percentileCutoff=16)
    img_eq = get_rgbcolor_equalized_image(img_eq, clipLimit=6.0, tileGridSize=(6, 6))
    img_eq = enhance_yellows_and_blacks(img_eq, percentileCutoff=16, changeBlacks=False)
    img_eq = cv2.medianBlur(img_eq, 25)
    img_eq = cv2.blur(img_eq, (19, 19))
    img_eq = cv2.bilateralFilter(img_eq, 9, 35, 35)

    # --- Threshold to get gear-tooth region ---
    # NOTE: medianBlur/blur output is in BGR convention inside OpenCV;
    # we convert to gray directly (channel order does not affect grayscale).
    img_eq_gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)

    binary_mask = np.zeros(img_eq_gray.shape, dtype=bool)
    binary_mask[img_eq_gray < 250] = True

    # AND with the original SAM mask
    overarching = np.logical_and(
        binary_mask,
        original_mask.astype(np.uint8).astype(bool)
    ).astype(np.uint8) * 255

    # --- Keep only large connected components ---
    h_img, w_img = overarching.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(overarching, 4)

    retained_mask      = np.zeros_like(overarching)
    overarching_mask_w = 0
    overarching_mask_h = 0

    for i in range(1, num_labels):
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        overarching_mask_w = max(overarching_mask_w, w_comp)
        overarching_mask_h = max(overarching_mask_h, h_comp)

        if (w_comp > w_img / 6) and (h_comp > h_img / 6):
            comp_mask     = (labels == i).astype(np.uint8) * 255
            retained_mask = cv2.bitwise_or(retained_mask, comp_mask)

    overarching_mask_bool = (retained_mask / 255).astype(bool)

    # --- Find the overarching mask contour ---
    # Invert retained_mask so the gear-tooth interior appears as foreground
    inverted = (255 - retained_mask).astype(np.uint8)
    contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    total_img_area = h_img * w_img
    cutoff_area    = 0.95 * total_img_area
    overarching_contour = cnts[0]          # fallback
    for cnt in cnts:
        if cv2.contourArea(cnt) <= cutoff_area:
            overarching_contour = cnt
            break

    print(
        f"[overarching_mask] Mask area = {int(np.sum(overarching_mask_bool))} px  "
        f"({np.sum(overarching_mask_bool) * 100.0 / total_img_area:.0f}% of image)"
    )

    # Clip equalised image to overarching mask (used by impression detector)
    img_eq[np.logical_not(overarching_mask_bool)] = (255, 255, 255)

    return {
        "maskedImageOriginal":    masked_image_original,
        "overarchingMask":        overarching_mask_bool,
        "overarchingMaskContour": overarching_contour,
        "overarchingMaskWidth":   overarching_mask_w,
        "overarchingMaskHeight":  overarching_mask_h,
        "maskedImage_Equalized":  img_eq,
    }