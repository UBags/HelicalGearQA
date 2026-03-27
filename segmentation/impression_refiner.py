"""
segmentation/impression_refiner.py
=====================================
SAM Pass 2 — refine the candidate impression regions using the centroids
from ``impression_detector`` as SAM prompt points.

The output of this module is a set of cleanly ordered, per-impression
data structures used by every downstream module:

  orderedFinalCentroidsOfImpressions
  orderedFinalMasksOfImpressions
  orderedFinalContoursOfImpressions
  orderedBoundsOfImpressions
  finalMask                               (combined mask, uint8 grayscale)
  maskForRemovingEdgesCloseToImpressions  (dilated finalMask)

Public function
---------------
refine_impressions
    Accepts the preFinalMask/centroids from the detector plus the SAM
    model components, and returns all the ordered structures above.
"""

from __future__ import annotations

import copy

import numpy as np
import cv2

from model.loader import SamComponents
from model.inference import run_prompted_inference_with_embeddings, get_image_embeddings
from utils.morphology_kernels import ellipticalKernel81_15


# ---------------------------------------------------------------------------
# Main refiner
# ---------------------------------------------------------------------------

def refine_impressions(pre_final_mask: np.ndarray,
                        final_centroids: list,
                        area_limits: list,
                        sam: SamComponents) -> dict:
    """
    Run SAM Pass 2 using impression centroids as prompt points, then
    clean, sort, and validate the resulting masks.

    Parameters
    ----------
    pre_final_mask : np.ndarray
        uint8 grayscale mask (H, W) from ``impression_detector.detect_impressions``.
        White pixels indicate candidate impression regions.
    final_centroids : list
        Centroid prompt points in SAM format, one per detected impression.
        Format: ``[ [[cx, cy]], [[cx, cy]], … ]``
        (will be wrapped into ``[final_centroids]`` for the processor).
    area_limits : list of [float, float]
        ±5 % area bands per centroid for mask validation.
    sam : SamComponents
        Loaded SAM model components.

    Returns
    -------
    dict with keys:
        ``orderedFinalCentroidsOfImpressions`` : list of [int, int]
        ``orderedFinalMasksOfImpressions``     : list of np.ndarray (bool H×W)
        ``orderedFinalContoursOfImpressions``  : list of np.ndarray (OpenCV contour)
        ``orderedBoundsOfImpressions``         : list of (x, y, w, h)
        ``finalMask``                          : np.ndarray (uint8 H×W)
        ``maskForRemovingEdgesCloseToImpressions`` : np.ndarray (uint8 H×W)

    Notes
    -----
    If no valid centroids were found by the detector (empty
    ``final_centroids``), all output lists will be empty and both mask
    arrays will be zero-filled.
    """
    # Shape of the working space (derived from pre_final_mask)
    h_img, w_img = pre_final_mask.shape

    # Empty outputs — returned when no centroids are available
    empty_result = {
        "orderedFinalCentroidsOfImpressions":       [],
        "orderedFinalMasksOfImpressions":           [],
        "orderedFinalContoursOfImpressions":        [],
        "orderedBoundsOfImpressions":               [],
        "finalMask":                                np.zeros((h_img, w_img), dtype=np.uint8),
        "maskForRemovingEdgesCloseToImpressions":   np.zeros((h_img, w_img), dtype=np.uint8),
    }

    if not final_centroids:
        print("[impression_refiner] No centroids found — skipping SAM Pass 2.")
        return empty_result

    # ------------------------------------------------------------------ #
    # SAM Pass 2 inference
    # ------------------------------------------------------------------ #
    # SAM expects the image as RGB; convert grayscale preFinalMask to RGB
    pre_final_rgb = cv2.cvtColor(pre_final_mask, cv2.COLOR_GRAY2RGB)

    # Wrap centroids into the nested format expected by SamProcessor:
    # [ [ [[cx1,cy1]], [[cx2,cy2]], … ] ]
    sam_input_points = [final_centroids]

    # Pre-compute embedding then run the prompted pass
    image_embeddings = get_image_embeddings(pre_final_rgb, sam)
    masks_list, scores = run_prompted_inference_with_embeddings(
        pre_final_rgb, sam_input_points, image_embeddings, sam
    )

    # ------------------------------------------------------------------ #
    # Select masks that match the expected area ranges
    # ------------------------------------------------------------------ #
    final_masks: list[np.ndarray] = []
    _, n_predictions, _ = scores.shape

    for pred_idx in range(n_predictions):
        found = False
        for mask_tensor in masks_list[0][pred_idx]:
            if found:
                break
            mask_np  = mask_tensor.cpu().detach().numpy()
            mask_area = float(np.sum(mask_np))
            for lo, hi in area_limits:
                if lo <= mask_area <= hi:
                    final_masks.append(mask_np)
                    found = True
                    break

    print(f"[impression_refiner] Chosen mask areas: {[int(np.sum(m)) for m in final_masks]}")

    # ------------------------------------------------------------------ #
    # Extract contours and centroids from accepted masks
    # ------------------------------------------------------------------ #
    contours:           list = []
    masks_to_delete:    list[int] = []
    centroids_raw:      list = []

    for idx, mask in enumerate(final_masks):
        try:
            mask_u8 = mask.astype(np.uint8) * 255
            n_contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Pick the longest contour when multiple are present
            best_ind = 0
            if len(n_contours) > 1:
                best_arc = 0.0
                for j, cnt in enumerate(n_contours):
                    arc = cv2.arcLength(cnt, True)
                    if arc > best_arc:
                        best_arc = arc
                        best_ind = j

            M  = cv2.moments(n_contours[best_ind])
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids_raw.append([cx, cy])
            contours.append(n_contours[best_ind])

        except Exception as exc:
            print(f"[impression_refiner] Contour extraction failed for mask {idx}: {exc}")
            masks_to_delete.append(idx)

    # Remove masks that failed contour extraction (in reverse order)
    for idx in reversed(masks_to_delete):
        final_masks.pop(idx)

    # ------------------------------------------------------------------ #
    # Build combined finalMask
    # ------------------------------------------------------------------ #
    final_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for mask in final_masks:
        final_mask = cv2.bitwise_or(final_mask, mask.astype(np.uint8) * 255)

    # ------------------------------------------------------------------ #
    # Sort centroids and build ordered per-impression lists
    # ------------------------------------------------------------------ #
    # Sort by (y, x) — top-to-bottom order in image coordinates
    sorted(centroids_raw, key=lambda k: [k[1], k[0]])
    print(f"[impression_refiner] Ordered centroids: {centroids_raw}")

    ordered_centroids:  list = []
    ordered_masks:      list = []
    ordered_contours:   list = []
    ordered_bounds:     list = []

    for centroid in centroids_raw:
        for i, mask in enumerate(final_masks):
            inside = cv2.pointPolygonTest(contours[i], tuple(centroid), False)
            if inside == 1:
                ordered_centroids.append(centroid)
                ordered_masks.append(final_masks[i])
                ordered_contours.append(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                ordered_bounds.append((x, y, w, h))

    # ------------------------------------------------------------------ #
    # Dilated mask for suppressing arc edges near impressions
    # ------------------------------------------------------------------ #
    mask_for_edges = cv2.morphologyEx(
        final_mask, cv2.MORPH_DILATE, ellipticalKernel81_15
    )

    return {
        "orderedFinalCentroidsOfImpressions":       ordered_centroids,
        "orderedFinalMasksOfImpressions":           ordered_masks,
        "orderedFinalContoursOfImpressions":        ordered_contours,
        "orderedBoundsOfImpressions":               ordered_bounds,
        "finalMask":                                final_mask,
        "maskForRemovingEdgesCloseToImpressions":   mask_for_edges,
    }