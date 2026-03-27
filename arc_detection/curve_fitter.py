"""
arc_detection/curve_fitter.py
==============================
Filter detected arc contours, fit quadratic polynomials, match each
fitted curve to its nearest impression contour, and produce the two
ordered sets of arc contours used for measurement.

Pipeline (mirrors the notebook exactly)
-----------------------------------------
1. ``filter_arc_contours``
   Remove arc contours that are too narrow (< 40 % of gear-tooth width)
   or whose quadratic fit coefficient a₀ < 0.0002 (too flat to be an arc).

2. ``match_arcs_to_impressions``
   For each impression contour, find the nearest arc contour using the
   minimum pairwise Euclidean distance (``scipy.spatial.distance.cdist``).
   Only the single closest arc per impression is kept.
   Returns ``closest_4_Contours``, ``closest_contour_distances``,
   ``closest_contour_centers``, and ``originalFittedCurves``.

3. ``build_ordered_arc_contours_1``
   Evaluate each fitted polynomial over the full image-width linspace to
   produce ``orderedChosenArcContours_1``.

4. ``translate_contour`` / ``closest_node``
   Helper: translate a contour so that its closest point to a target
   node is exactly at that node (used to centre arcs on impression
   centroids before the second polyfit).

5. ``build_ordered_arc_contours_2_and_intersections``
   For each impression centroid, translate the corresponding
   ``orderedChosenArcContours_1`` entry, re-fit a polynomial, evaluate
   over the linspace, draw the curve on the overarching-mask image,
   and compute the two intersection points with the gear-boundary contour
   using K-Means(k=2) on the intersection pixel coordinates.
   Returns ``orderedChosenArcContours_2``, ``intersectionPoints``,
   ``finalFittedCurves``, and ``edgeAndContactAndContour`` (display image).

6. ``filter_impressions_by_intersection``
   Remove impressions whose two intersection points are closer than
   overarchingMaskWidth / 3 (arc does not span the full gear face).

7. ``filter_impressions_by_width``
   Remove impressions whose contact width is less than 75 % of the
   average (likely an outlier or mis-detected impression).

8. ``filter_impressions_by_spacing``
   When more than 2 impressions remain, remove any whose Y-spacing
   relative to neighbours deviates by more than 20 % from the minimum
   spacing (evenly-spaced impressions are expected).
"""

from __future__ import annotations

import copy

import numpy as np
import cv2
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import k_means

from measurement.width_calculator import get_max_width_and_location


# ---------------------------------------------------------------------------
# Helper: sorted insertion
# ---------------------------------------------------------------------------

def _find_fitment_index(new_element: float, element_array: list) -> int:
    """Return the index at which *new_element* should be inserted to keep
    *element_array* sorted in ascending order."""
    for i, element in enumerate(element_array):
        if new_element < element:
            return i
    return len(element_array)


# ---------------------------------------------------------------------------
# Helper: translate contour
# ---------------------------------------------------------------------------

def closest_node(node: np.ndarray,
                  contour) -> tuple[np.ndarray, int]:
    """
    Return the point in *contour* that is closest to *node* and its index.

    Parameters
    ----------
    node : array-like, shape (2,)
        Target point [x, y].
    contour : array-like
        Contour in ``[points]`` or ``[[x,y], …]`` format.

    Returns
    -------
    closest_point : np.ndarray, shape (2,)
    index : int
    """
    squeezed = np.squeeze(contour)
    idx = distance.cdist([node], squeezed).argmin()
    return squeezed[idx], int(idx)


def translate_contour(node: np.ndarray, contour) -> np.ndarray:
    """
    Translate *contour* so that its closest point to *node* lies exactly
    on *node*.

    Parameters
    ----------
    node : array-like, shape (2,)  [x, y]
    contour : array-like (SAM / polylines format)

    Returns
    -------
    np.ndarray
        Translated contour in ``[points]`` format (shape (1, N, 2)).
    """
    closest, _ = closest_node(node, contour)
    squeezed    = np.squeeze(contour)
    dx = node[0] - closest[0]
    dy = node[1] - closest[1]
    new_pts = squeezed + np.array([dx, dy])
    return np.array([new_pts])


# ---------------------------------------------------------------------------
# Step 1 — filter arc contours
# ---------------------------------------------------------------------------

def filter_arc_contours(arc_contours: list,
                         overarching_mask_width: int) -> list:
    """
    Remove arc contours that are too narrow or too flat.

    Parameters
    ----------
    arc_contours : list
        Raw OpenCV contours from ``cv2.findContours`` on the arc-edge image.
    overarching_mask_width : int
        Width of the gear-tooth bounding box in pixels.

    Returns
    -------
    list
        Filtered list of contours.
    """
    # Pass 1: width filter (< 40 % of gear width)
    filtered = []
    for cnt in arc_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= overarching_mask_width * 0.4:
            filtered.append(cnt)

    # Pass 2: curvature filter (a₀ < 0.0002 → too straight)
    curved = []
    for cnt in filtered:
        pts = cnt[:, 0, :]
        x_pts = pts[:, 0].astype(float)
        y_pts = pts[:, 1].astype(float)
        coeffs = np.polyfit(x_pts, y_pts, 2)
        if coeffs[0] >= 0.0002:
            curved.append(cnt)

    print(f"[curve_fitter] Viable arc contours after filtering: {len(curved)}")
    return curved


# ---------------------------------------------------------------------------
# Step 2 — match arcs to impressions
# ---------------------------------------------------------------------------

def match_arcs_to_impressions(arc_contours: list,
                                ordered_impression_contours: list) -> dict:
    """
    Match each impression contour to its nearest arc contour using
    minimum pairwise Euclidean distance.

    Only the single closest arc per impression is retained (the notebook
    comment reads: "Code modified to add only the nearest contour").

    Parameters
    ----------
    arc_contours : list
        Filtered arc contours (from ``filter_arc_contours``).
    ordered_impression_contours : list
        Ordered impression contours from ``impression_refiner``.

    Returns
    -------
    dict with keys:
        ``closest_4_Contours``     : list of np.ndarray  (one per impression)
        ``closest_contour_distances`` : list of float
        ``closest_contour_centers``   : list of (float, float)
        ``originalFittedCurves``      : list of np.ndarray  (polyfit coeffs)
    """
    closest_contours:   list = []
    closest_distances:  list = []
    closest_centers:    list = []

    for impression_cnt in ordered_impression_contours:
        impression_pts = np.squeeze(impression_cnt)
        close_cnts:    list = []
        close_dists:   list = []
        close_ctrs:    list = []

        for arc_cnt in arc_contours:
            arc_pts = np.squeeze(arc_cnt)
            cx = float(np.mean(arc_pts[:, 0]))
            cy = float(np.mean(arc_pts[:, 1]))

            dist_mat = distance.cdist(impression_pts, arc_pts, 'euclidean')
            min_dist = float(np.min(dist_mat))

            if len(close_cnts) == 0:
                close_cnts.append(arc_cnt)
                close_dists.append(min_dist)
                close_ctrs.append((cx, cy))
            elif len(close_cnts) < 4:
                fi = _find_fitment_index(min_dist, close_dists)
                close_cnts.insert(fi, arc_cnt)
                close_dists.insert(fi, min_dist)
                close_ctrs.insert(fi, (cx, cy))
            else:
                fi = _find_fitment_index(min_dist, close_dists)
                if fi < 4:
                    close_cnts.insert(fi, arc_cnt)
                    close_dists.insert(fi, min_dist)
                    close_ctrs.insert(fi, (cx, cy))
                    del close_cnts[-1]
                    del close_dists[-1]
                    del close_ctrs[-1]

        # Only the nearest contour is used
        if close_cnts:
            closest_contours.append(close_cnts[0])
            closest_distances.append(close_dists[0])
            closest_centers.append(close_ctrs[0])

    # Fit polynomial to each matched arc contour
    original_fitted_curves: list = []
    for cnt in closest_contours:
        pts = cnt[:, 0, :]
        x_pts = pts[:, 0].astype(float)
        y_pts = pts[:, 1].astype(float)
        original_fitted_curves.append(np.polyfit(x_pts, y_pts, 2))

    return {
        "closest_4_Contours":          closest_contours,
        "closest_contour_distances":   closest_distances,
        "closest_contour_centers":     closest_centers,
        "originalFittedCurves":        original_fitted_curves,
    }


# ---------------------------------------------------------------------------
# Step 3 — build orderedChosenArcContours_1
# ---------------------------------------------------------------------------

def build_ordered_arc_contours_1(original_fitted_curves: list,
                                   image_width: int) -> tuple[list, np.ndarray]:
    """
    Evaluate each polynomial over a full-width linspace and return the
    resulting point arrays as ``orderedChosenArcContours_1``.

    Parameters
    ----------
    original_fitted_curves : list of np.ndarray
        Polynomial coefficients (a, b, c) for each matched arc.
    image_width : int
        Width of the arc-detection image in pixels.

    Returns
    -------
    ordered_arc_contours_1 : list of [np.ndarray]
        One entry per impression; each entry is a list containing a
        (W, 2) integer array of (x, y) curve points.
    lspace : np.ndarray
        The linspace array (shape (image_width,)) shared by all curves.
    """
    lspace = np.linspace(0, image_width - 1, image_width)
    ordered: list = []
    for coeffs in original_fitted_curves:
        y_fit = coeffs[0] * lspace ** 2 + coeffs[1] * lspace + coeffs[2]
        pts   = np.array(list(zip(lspace.astype(int), y_fit.astype(int))))
        ordered.append([pts])
    return ordered, lspace


# ---------------------------------------------------------------------------
# Step 5 — build orderedChosenArcContours_2 + intersection points
# ---------------------------------------------------------------------------

def build_ordered_arc_contours_2_and_intersections(
        ordered_centroids: list,
        ordered_arc_contours_1: list,
        ordered_impression_masks: list,
        overarching_mask_image: np.ndarray,
        overarching_mask_width: int,
        arc_edge_image_shape: tuple,
        lspace: np.ndarray) -> dict:
    """
    Translate each arc to pass through its impression centroid, re-fit,
    compute where it intersects the gear-boundary contour image, and
    identify the two boundary-crossing points via K-Means.

    Parameters
    ----------
    ordered_centroids : list of [int, int]
        Ordered impression centroids from ``impression_refiner``.
    ordered_arc_contours_1 : list of [np.ndarray]
        First-pass arc contours from ``build_ordered_arc_contours_1``.
    ordered_impression_masks : list of np.ndarray
        Ordered boolean impression masks.
    overarching_mask_image : np.ndarray
        Binary image (H, W) uint8 with the gear-boundary contour drawn
        at thickness=3 (used for intersection detection).
    overarching_mask_width : int
        Width of the gear-tooth bounding box in pixels.
    arc_edge_image_shape : tuple
        Shape (H, W) of the arc-edge image.
    lspace : np.ndarray
        Linspace array shared by all curves.

    Returns
    -------
    dict with keys:
        ``orderedChosenArcContours_2``  : list of np.ndarray
        ``intersectionPoints``         : list of (np.ndarray, np.ndarray)
        ``finalFittedCurves``          : list of np.ndarray  (coefficients)
        ``impressionsToBeDeleted``     : list of int  (bad intersection indices)
        ``edgeAndContactAndContour``   : np.ndarray  (uint8 display image)
        ``overarchingMaskImageDisplay``: np.ndarray  (uint8 display image)
    """
    h, w = arc_edge_image_shape

    ordered_contours_2:       list = []
    intersection_points:      list = []
    final_fitted_curves:      list = []
    impressions_to_delete:    list = []
    oam_display = copy.deepcopy(overarching_mask_image)

    for i, centroid in enumerate(ordered_centroids):
        new_contour = translate_contour(centroid, ordered_arc_contours_1[i])

        pts    = np.squeeze(new_contour)
        x_pts  = pts[:, 0].astype(float)
        y_pts  = pts[:, 1].astype(float)
        coeffs = np.polyfit(x_pts, y_pts, 2)
        final_fitted_curves.append(coeffs)

        y_fit     = coeffs[0] * lspace ** 2 + coeffs[1] * lspace + coeffs[2]
        new_points = np.array(list(zip(lspace.astype(int), y_fit.astype(int))))

        ordered_contours_2.append(new_contour)

        # Draw the translated curve on the contour image
        contour_img = np.zeros((h, w), np.uint8)
        cv2.polylines(contour_img, [new_points], False, 255, thickness=2)
        cv2.polylines(oam_display,  [new_points], False, 255, thickness=2)

        # Intersection of curve with gear boundary
        oam_thresh = copy.deepcopy(overarching_mask_image)
        oam_thresh[oam_thresh > 128] = 255
        contour_img[contour_img > 128] = 255
        intersection_img = cv2.bitwise_and(oam_thresh, contour_img)
        intersection_bool = (intersection_img / 255).astype(bool)
        coords = np.argwhere(intersection_bool)

        # Remove edge pixels (within 4 px of image border)
        valid_indices = [
            j for j, pt in enumerate(coords)
            if pt[0] > 4 and pt[0] < h - 5 and pt[1] > 4 and pt[1] < w - 5
        ]
        coords = coords[valid_indices] if valid_indices else coords

        if coords.shape[0] >= 2:
            df   = pd.DataFrame(coords)
            km   = k_means(df, n_clusters=2, n_init='auto')
            p1   = km[0][0]
            p2   = km[0][1]
            dist = float(np.linalg.norm(p1 - p2))

            if dist < overarching_mask_width / 3:
                # Intersection points too close — arc does not span the gear
                intersection_points.append(
                    (p2.astype(np.int32), p1.astype(np.int32))
                )
                impressions_to_delete.append(i)
            elif p1[1] > p2[1]:
                intersection_points.append(
                    (p2.astype(np.int32), p1.astype(np.int32))
                )
            else:
                intersection_points.append(
                    (p1.astype(np.int32), p2.astype(np.int32))
                )

    # Build edge-and-contact display image
    edge_and_contact = copy.deepcopy(oam_display)
    for mask in ordered_impression_masks:
        edge_and_contact[mask] = 255

    return {
        "orderedChosenArcContours_2":   ordered_contours_2,
        "intersectionPoints":           intersection_points,
        "finalFittedCurves":            final_fitted_curves,
        "impressionsToBeDeleted":       impressions_to_delete,
        "edgeAndContactAndContour":     edge_and_contact,
        "overarchingMaskImageDisplay":  oam_display,
    }


# ---------------------------------------------------------------------------
# Steps 6-8 — impression pruning helpers
# ---------------------------------------------------------------------------

def _pop_all(idx: int, *lists: list) -> None:
    """Pop index *idx* from every list in *lists* (in place)."""
    for lst in lists:
        try:
            lst.pop(idx)
        except IndexError:
            pass


def filter_impressions_by_intersection(
        impressions_to_delete: list,
        ordered_centroids:      list,
        ordered_masks:          list,
        ordered_contours:       list,
        ordered_bounds:         list,
        closest_contours:       list,
        closest_distances:      list,
        closest_centers:        list,
        original_fitted_curves: list,
        arc_contours_1:         list,
        arc_contours_2:         list,
        final_fitted_curves:    list,
        intersection_points:    list) -> None:
    """
    Remove impressions (in place) whose arcs do not properly intersect
    the gear boundary (i.e. the two intersection points are too close).

    All lists are mutated directly; nothing is returned.
    """
    for i in reversed(impressions_to_delete):
        _pop_all(i,
                 ordered_centroids, ordered_masks, ordered_contours,
                 ordered_bounds, closest_contours, closest_distances,
                 closest_centers, original_fitted_curves,
                 arc_contours_1, arc_contours_2, final_fitted_curves,
                 intersection_points)
    impressions_to_delete.clear()


def filter_impressions_by_width(
        ordered_centroids:      list,
        ordered_masks:          list,
        ordered_contours:       list,
        ordered_bounds:         list,
        closest_contours:       list,
        closest_distances:      list,
        closest_centers:        list,
        original_fitted_curves: list,
        arc_contours_1:         list,
        arc_contours_2:         list,
        final_fitted_curves:    list,
        intersection_points:    list) -> None:
    """
    Remove impressions whose raw pixel width is less than 75 % of the
    average width across all impressions.

    All lists are mutated in place.
    """
    raw_widths: list[float] = []
    for mask in ordered_masks:
        _, max_w, _ = get_max_width_and_location(copy.deepcopy(mask))
        raw_widths.append(float(max_w))

    if len(raw_widths) > 1:
        sorted_w = sorted(raw_widths)[1:]   # drop the smallest (likely outlier)
    else:
        sorted_w = list(raw_widths)

    avg_w = float(np.mean(sorted_w)) if sorted_w else 0.0

    to_delete = [i for i, w in enumerate(raw_widths) if w < 0.75 * avg_w]
    for i in reversed(to_delete):
        _pop_all(i,
                 ordered_centroids, ordered_masks, ordered_contours,
                 ordered_bounds, closest_contours, closest_distances,
                 closest_centers, original_fitted_curves,
                 arc_contours_1, arc_contours_2, final_fitted_curves,
                 intersection_points)


def filter_impressions_by_spacing(
        ordered_centroids:      list,
        ordered_masks:          list,
        ordered_contours:       list,
        ordered_bounds:         list,
        closest_contours:       list,
        closest_distances:      list,
        closest_centers:        list,
        original_fitted_curves: list,
        arc_contours_1:         list,
        arc_contours_2:         list,
        final_fitted_curves:    list,
        intersection_points:    list) -> None:
    """
    When more than 2 impressions remain, remove any whose Y-spacing
    relative to their neighbours deviates by more than 20 % from the
    minimum spacing (evenly-spaced impressions are expected on a gear).

    All lists are mutated in place.
    """
    if len(ordered_centroids) <= 2:
        return

    y_vals    = [c[1] for c in ordered_centroids]
    gaps      = [y_vals[i + 1] - y_vals[i] for i in range(len(y_vals) - 1)]
    min_gap   = min(gaps)
    multiples = [g / min_gap for g in gaps]

    print(f"[curve_fitter] Y spacing multiples: {multiples}")

    ok_indices: list[int] = []
    for i, mult in enumerate(multiples):
        if 0.8 < mult < 1.2:
            ok_indices.append(i)
            ok_indices.append(i + 1)
    ok_indices = sorted(set(ok_indices))

    to_delete = [
        i for i in range(len(ordered_centroids)) if i not in ok_indices
    ]
    print(f"[curve_fitter] Spacing filter removing indices: {to_delete}")

    for i in reversed(to_delete):
        _pop_all(i,
                 ordered_centroids, ordered_masks, ordered_contours,
                 ordered_bounds, closest_contours, closest_distances,
                 closest_centers, original_fitted_curves,
                 arc_contours_1, arc_contours_2, final_fitted_curves,
                 intersection_points)