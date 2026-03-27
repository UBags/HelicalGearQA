"""
measurement/width_calculator.py
================================
Compute the maximum width (contact-impression thickness) of a binary
mask by rotating the mask to its principal axis, finding the widest
row in the rectified image, and returning the pixel count.

This faithfully implements ``getMaxWidthAndLocationOfMaxWidth`` from the
original notebook.

Public function
---------------
get_max_width_and_location(mask) → (max_row_index, max_width, rectified)
"""

from __future__ import annotations

import numpy as np
import cv2


def get_max_width_and_location(mask: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Find the maximum horizontal width of the foreground region in *mask*
    after rotating it to its minimum-area bounding-rectangle orientation.

    Algorithm
    ---------
    1. Threshold the mask to binary (0/255).
    2. Find the single largest external contour.
    3. Compute ``cv2.minAreaRect`` and derive the de-rotation angle.
    4. Warp-affine the image to un-rotate ("rectify") it.
    5. Re-threshold, keep only the largest connected component, then
       find its bounding box (trimming the top/bottom 5 % to avoid edge
       artefacts).
    6. Count non-zero pixels in each row of the cropped region and return
       the row with the maximum count.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or uint8 binary mask, shape (H, W).

    Returns
    -------
    max_row_index : int
        Row index (within the cropped bounding box) of the widest row.
        Returns -1 if no foreground pixels are found.
    max_width : int
        Maximum number of foreground pixels in any single row.
        Returns 0 if no foreground pixels are found.
    rectified : np.ndarray
        The de-rotated, thresholded uint8 image (H, W).
    """
    hh, ww = mask.shape[:2]
    mask_u8 = mask.astype(np.uint8)
    gray    = (mask_u8 * 255).astype(np.uint8)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if not contours:
        return -1, 0, thresh

    big_contour = max(contours, key=cv2.contourArea)

    # --- De-rotation ---
    rotrect             = cv2.minAreaRect(big_contour)
    (center), (width, height), angle = rotrect

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -(90 + angle) if width > height else -angle

    M         = cv2.getRotationMatrix2D(center, -angle, scale=1.0)
    rectified = cv2.warpAffine(
        gray, M, (ww, hh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    rectified = rectified.astype(np.uint8)
    _, rectified = cv2.threshold(rectified, 127, 255, cv2.THRESH_BINARY)

    # --- Keep only the largest connected component ---
    detect_mask = np.zeros(rectified.shape, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rectified, 4)

    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    if not areas:
        return -1, 0, rectified

    max_area = max(areas)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 0.95 * max_area:
            comp_mask   = (labels == i).astype(np.uint8) * 255
            detect_mask = cv2.bitwise_or(detect_mask, comp_mask)

    # --- Bounding box + row widths ---
    cntrs = cv2.findContours(detect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    if not cntrs:
        return -1, 0, rectified

    cntr         = cntrs[0]
    x, y, w, h   = cv2.boundingRect(cntr)

    # Trim top/bottom 5 % to exclude noisy edges
    y_trim = y + h // 20
    h_trim = 19 * (h // 20)
    crop   = detect_mask[y_trim: y_trim + h_trim, x: x + w]

    row_counts = np.count_nonzero(crop, axis=1)
    if len(row_counts) == 0:
        return -1, 0, rectified

    max_row   = int(np.argmax(row_counts))
    max_width = int(row_counts[max_row])
    return max_row, max_width, rectified