"""
measurement/arc_length.py
==========================
Arc-length integration and impression-projection helpers.

Functions
---------
arc_length_integral
    Compute the arc length of a curve y=f(x) between two x-bounds using
    the trapezoid rule on the gradient of y w.r.t. x.

get_impression_projections_on_arc
    Find the left-most and right-most points of an impression contour
    projected onto the nearest points of an arc contour.

get_curve_subset
    Extract the sub-arc between two boundary points.

compute_measurements
    Orchestrate the per-impression measurement loop:
    for each impression, compute toe clearance, contact length, heel
    clearance, and contact width in pixel units, then scale to mm using
    the known physical arc length of the component.
"""

from __future__ import annotations

import copy

import numpy as np
import cv2
from scipy.spatial import distance

from measurement.width_calculator import get_max_width_and_location


# ---------------------------------------------------------------------------
# Arc-length integral
# ---------------------------------------------------------------------------

def arc_length_integral(x: np.ndarray,
                         y: np.ndarray,
                         a: float,
                         b: float) -> float:
    """
    Compute the arc length of the curve defined by arrays *x* and *y*
    between *x* = *a* and *x* = *b* using the trapezoidal rule.

    Arc length = ∫ₐᵇ √(1 + (dy/dx)²) dx

    Parameters
    ----------
    x : np.ndarray
        Monotonically increasing x-coordinates of the curve.
    y : np.ndarray
        y-coordinates corresponding to each *x* value.
    a, b : float
        Lower and upper x-bounds for integration.

    Returns
    -------
    float
        Approximate arc length in the same units as *x* and *y* (pixels).
        Returns 0.0 if the bounded region contains fewer than 2 points.
    """
    bounds = (x >= a) & (x <= b)
    x_b = x[bounds]
    y_b = y[bounds]
    if x_b.size < 2:
        return 0.0
    return float(
        np.trapz(
            np.sqrt(1 + np.gradient(y_b, x_b) ** 2),
            x_b, dx=0.1, axis=-1
        )
    )


# ---------------------------------------------------------------------------
# Impression projection
# ---------------------------------------------------------------------------

def get_impression_projections_on_arc(
        impression_contour: np.ndarray,
        arc_contour) -> tuple[np.ndarray, np.ndarray]:
    """
    Project the leftmost and rightmost points of *impression_contour*
    onto their nearest neighbours on *arc_contour*.

    Parameters
    ----------
    impression_contour : np.ndarray
        OpenCV contour of the impression, shape (N, 1, 2).
    arc_contour : array-like
        Arc contour points, shape (M, 2) or (1, M, 2).

    Returns
    -------
    left_point : np.ndarray, shape (2,)
        Point on *arc_contour* nearest to the leftmost impression point.
    right_point : np.ndarray, shape (2,)
        Point on *arc_contour* nearest to the rightmost impression point.
    """
    left_raw  = tuple(impression_contour[impression_contour[:, :, 0].argmin()][0])
    right_raw = tuple(impression_contour[impression_contour[:, :, 0].argmax()][0])

    arc_pts = np.squeeze(arc_contour)
    left_idx  = distance.cdist([left_raw],  arc_pts).argmin()
    right_idx = distance.cdist([right_raw], arc_pts).argmin()

    print(
        f"  [arc_length] left={left_raw}, right={right_raw}, "
        f"left_idx={left_idx}, right_idx={right_idx}"
    )
    return arc_pts[left_idx], arc_pts[right_idx]


# ---------------------------------------------------------------------------
# Curve subset extraction
# ---------------------------------------------------------------------------

def get_curve_subset(starting_node: np.ndarray,
                      ending_node:   np.ndarray,
                      contour) -> tuple[list, np.ndarray]:
    """
    Return the portion of *contour* between the points closest to
    *starting_node* and *ending_node*.

    Note: coordinates are expected in (row, col) order (as returned by
    ``np.argwhere``), so they are reversed to (col, row) = (x, y) for
    the ``distance.cdist`` look-up.

    Parameters
    ----------
    starting_node : np.ndarray, shape (2,)
        Start point in (row, col) = (y, x) order.
    ending_node : np.ndarray, shape (2,)
        End point in (row, col) = (y, x) order.
    contour : array-like
        Arc contour in ``[points]`` format, shape (1, N, 2) with
        columns (x, y).

    Returns
    -------
    new_contour : list
        ``[np.ndarray]`` where the array has shape (K, 2), suitable for
        ``cv2.polylines``.
    squeezed_subset : np.ndarray
        Raw (K, 2) sub-array of the contour.
    """
    # Intersection points are (row, col); flip to (x, y) = (col, row)
    start_xy = np.array([starting_node[1], starting_node[0]])
    end_xy   = np.array([ending_node[1],   ending_node[0]])

    squeezed = np.squeeze(contour)
    start_idx = int(distance.cdist([start_xy], squeezed).argmin())
    end_idx   = int(distance.cdist([end_xy],   squeezed).argmin())

    subset = copy.deepcopy(squeezed[start_idx: end_idx + 1, :])
    new_contour = [subset]
    return new_contour, subset


# ---------------------------------------------------------------------------
# Main measurement loop
# ---------------------------------------------------------------------------

def compute_measurements(ordered_arc_contours_2: list,
                           ordered_impression_contours: list,
                           ordered_impression_masks: list,
                           intersection_points: list,
                           arc_length_mm: float,
                           width_multiplier: float,
                           overarching_mask: np.ndarray,
                           final_mask: np.ndarray,
                           image_for_final_display: np.ndarray) -> dict:
    """
    Compute all five gear-tooth measurements for the current image.

    For each impression *i*:
      - Crop the arc between its two boundary intersection points.
      - Integrate arc length over three segments: left gap (toe), contact
        zone, right gap (heel).
      - Measure the maximum cross-sectional width of the impression mask.
      - Scale all pixel measurements to mm using *arc_length_mm*.

    Final values are averaged across all impressions.

    Parameters
    ----------
    ordered_arc_contours_2 : list
        Translated and re-fitted arc contours (one per impression).
    ordered_impression_contours : list
        Ordered OpenCV contours of the impressions.
    ordered_impression_masks : list
        Ordered boolean impression masks.
    intersection_points : list of (np.ndarray, np.ndarray)
        Boundary intersection point pairs (one per impression).
    arc_length_mm : float
        Known physical arc length of this component in mm.
    width_multiplier : float
        Scaling factor applied to the raw pixel width measurement.
    overarching_mask : np.ndarray
        Boolean array (H, W) — True inside the gear-tooth region.
    final_mask : np.ndarray
        Combined impression mask (H, W), uint8.
    image_for_final_display : np.ndarray
        RGB image that measurement overlays (arcs, tick marks) are drawn
        onto *in place*.

    Returns
    -------
    dict with keys:
        ``toe_clearance_mm``   : float
        ``contact_length_mm``  : float
        ``heel_clearance_mm``  : float
        ``contact_width_mm``   : float
        ``image_for_final_display`` : np.ndarray  (with overlays drawn)
        ``image_to_be_saved``  : np.ndarray  (finalMask RGB + arc overlays)
    """
    required_arc_lengths: list[list[float]] = [[], [], []]
    required_widths:      list[float]       = []

    # Build the RGB overlay image from finalMask
    img_overlay = copy.deepcopy(final_mask)
    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_GRAY2RGB)

    for i, arc_contour in enumerate(ordered_arc_contours_2):
        print(f"[arc_length] Processing contour {i}")
        cv2.polylines(img_overlay,      arc_contour, False, (0, 255, 255), thickness=6)
        img_overlay[~overarching_mask] = (255, 255, 255)

        try:
            start_pt = intersection_points[i][0]
            end_pt   = intersection_points[i][1]

            new_contour, _ = get_curve_subset(start_pt, end_pt, arc_contour)
            cv2.polylines(img_overlay,             new_contour, False, (255, 0, 0), thickness=2)
            cv2.polylines(image_for_final_display, new_contour, False, (255, 0, 0), thickness=2)

            # Draw intersection dots
            for pt_pair in intersection_points:
                for pt in pt_pair:
                    cv2.circle(img_overlay,             (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)
                    cv2.circle(image_for_final_display, (int(pt[1]), int(pt[0])), 4, (255, 0, 0), -1)

            xy = new_contour[0]
            X, Y = np.split(xy, [-1], axis=1)
            X = np.squeeze(X).astype(float)
            Y = np.squeeze(Y).astype(float)

            impression_cnt = ordered_impression_contours[i]
            left, right = get_impression_projections_on_arc(impression_cnt, arc_contour)

            try:
                left_gap = arc_length_integral(X, Y, X[0],    float(left[0]))
            except Exception:
                left_gap = 0.0
            try:
                mid_imp  = arc_length_integral(X, Y, float(left[0]),  float(right[0]))
            except Exception:
                mid_imp  = 0.0
            try:
                right_gap = arc_length_integral(X, Y, float(right[0]), X[-1])
            except Exception:
                right_gap = 0.0

            total_px = left_gap + mid_imp + right_gap
            if total_px > 0:
                scale = arc_length_mm / total_px
            else:
                scale = 0.0

            required_arc_lengths[0].append(round(left_gap  * scale, 1))
            required_arc_lengths[1].append(round(mid_imp   * scale, 1))
            required_arc_lengths[2].append(round(right_gap * scale, 1))

            current_mask = ordered_impression_masks[i]
            _, max_width_px, _ = get_max_width_and_location(copy.deepcopy(current_mask))
            max_width_mm = round(max_width_px * scale, 1)
            required_widths.append(max_width_mm)

            # Green tick marks at impression left/right projections
            for img in (img_overlay, image_for_final_display):
                cv2.line(img, (int(left[0]),  int(left[1])  - 8),
                              (int(left[0]),  int(left[1])  + 8), (0, 255, 127), 6)
                cv2.line(img, (int(right[0]), int(right[1]) - 8),
                              (int(right[0]), int(right[1]) + 8), (0, 255, 127), 6)

        except Exception as exc:
            print(f"[arc_length] Contour {i}: {exc}")

    # Average across impressions
    def _safe_mean(lst: list) -> float:
        return round(float(np.mean(lst)), 1) if lst else 0.0

    toe_clearance_mm  = _safe_mean(required_arc_lengths[0])
    contact_length_mm = _safe_mean(required_arc_lengths[1])
    heel_clearance_mm = _safe_mean(required_arc_lengths[2])
    contact_width_mm  = round(
        _safe_mean(required_widths) * width_multiplier, 1
    )

    print(
        f"[arc_length] Toe={toe_clearance_mm} mm  Contact={contact_length_mm} mm  "
        f"Heel={heel_clearance_mm} mm  Width={contact_width_mm} mm"
    )

    return {
        "toe_clearance_mm":          toe_clearance_mm,
        "contact_length_mm":         contact_length_mm,
        "heel_clearance_mm":         heel_clearance_mm,
        "contact_width_mm":          contact_width_mm,
        "image_for_final_display":   image_for_final_display,
        "image_to_be_saved":         img_overlay,
    }