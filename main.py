"""
main.py
========
Top-level entry point for the gear-inspection pipeline.

Run this script from the project root in PyCharm (or any terminal):

    python main.py

The script:
  1. Loads the SAM model once.
  2. Iterates over every image file in ``config.settings.IMAGES_DIRECTORY``.
  3. For each image, runs the full pipeline:
       load → SAM Pass 1 → overarching mask → impression detection
       → SAM Pass 2 → arc extraction → curve fitting → measurements
       → display
  4. Prints the five measurements to stdout and displays the three-panel
     matplotlib figure.

All configurable parameters (paths, prompt points, tolerances) live in
``config/settings.py``.
"""

import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

# --- Configuration ---
from config.settings import IMAGES_DIRECTORY, DIMENSIONS_TABLE_HEIGHT

# --- Utilities ---
from utils.image_io import (
    load_image_rgb,
    get_input_points,
    get_arc_length_mm,
    get_arc_length_tolerances,
    get_width_multiplier,
    iter_image_files,
)

# --- Model ---
from model.loader import load_sam_components
from model.inferencer import run_prompted_inference

# --- Segmentation ---
from segmentation.overarching_mask import select_best_sam_mask, compute_overarching_mask
from segmentation.impression_detector import detect_impressions
from segmentation.impression_refiner import refine_impressions

# --- Arc detection ---
from arc_detection.edge_extractor import extract_arc_edges
from arc_detection.curve_fitter import (
    filter_arc_contours,
    match_arcs_to_impressions,
    build_ordered_arc_contours_1,
    build_ordered_arc_contours_2_and_intersections,
    filter_impressions_by_intersection,
    filter_impressions_by_width,
    filter_impressions_by_spacing,
)

# --- Measurement ---
from measurement.arc_length import compute_measurements

# --- Display ---
from display.results_table import (
    make_dimensions_table,
    assemble_final_image,
    overlay_image_alpha,
)
from display.visualiser import render_results


# ---------------------------------------------------------------------------
# Single-image pipeline
# ---------------------------------------------------------------------------

def process_image(image_path: str, sam) -> None:
    """
    Run the complete gear-inspection pipeline on one image.

    Parameters
    ----------
    image_path : str
        Full path to the input image file.
    sam : SamComponents
        Pre-loaded SAM model components (from ``model.loader``).
    """
    filename  = Path(image_path).name
    t_start   = datetime.now()
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Load image and per-component configuration
    # ------------------------------------------------------------------ #
    raw_image   = load_image_rgb(image_path)
    inp_points  = get_input_points(filename)
    arc_len_mm  = get_arc_length_mm(filename)
    tolerances  = get_arc_length_tolerances(filename)
    width_mult  = get_width_multiplier(filename)

    print(f"Arc length: {arc_len_mm} mm  |  Width multiplier: {width_mult}")

    if inp_points is None:
        print("No prompt points found for this image — skipping.")
        return

    # ------------------------------------------------------------------ #
    # 2. SAM Pass 1 — generate prompted masks
    # ------------------------------------------------------------------ #
    print("Running SAM Pass 1 …")
    masks_list, scores = run_prompted_inference(raw_image, inp_points, sam)

    # ------------------------------------------------------------------ #
    # 3. Select best mask and derive the overarching gear-tooth region
    # ------------------------------------------------------------------ #
    masked_image, original_mask = select_best_sam_mask(raw_image, masks_list, scores)

    oam = compute_overarching_mask(masked_image, original_mask)
    masked_image_original  = oam["maskedImageOriginal"]
    overarching_mask       = oam["overarchingMask"]
    overarching_contour    = oam["overarchingMaskContour"]
    overarching_mask_w     = oam["overarchingMaskWidth"]
    overarching_mask_h     = oam["overarchingMaskHeight"]
    masked_image_equalized = oam["maskedImage_Equalized"]

    # ------------------------------------------------------------------ #
    # 4. Impression detection (Chan-Vese → yellow/black → morphological)
    # ------------------------------------------------------------------ #
    print("Detecting impressions …")
    det = detect_impressions(
        masked_image,
        masked_image_equalized,
        overarching_mask,
        overarching_mask_w,
        overarching_mask_h,
    )
    pre_final_mask        = det["preFinalMask"]
    final_centroids       = det["finalCentroids"]
    area_limits           = det["areaLimits"]
    enhanced_yb_image     = det["enhancedYellowsAndBlacksImage"]

    # ------------------------------------------------------------------ #
    # 5. SAM Pass 2 — refine impression masks
    # ------------------------------------------------------------------ #
    print("Running SAM Pass 2 (impression refinement) …")
    ref = refine_impressions(pre_final_mask, final_centroids, area_limits, sam)

    ordered_centroids  = ref["orderedFinalCentroidsOfImpressions"]
    ordered_masks      = ref["orderedFinalMasksOfImpressions"]
    ordered_contours   = ref["orderedFinalContoursOfImpressions"]
    ordered_bounds     = ref["orderedBoundsOfImpressions"]
    final_mask         = ref["finalMask"]
    mask_for_edges     = ref["maskForRemovingEdgesCloseToImpressions"]

    if not ordered_masks:
        print("No impressions detected — skipping measurement.")
        return

    # ------------------------------------------------------------------ #
    # 6. Arc-edge extraction
    # ------------------------------------------------------------------ #
    print("Extracting arc edges …")
    arc_edge_image = extract_arc_edges(
        masked_image_original,
        overarching_mask,
        mask_for_edges,
        ordered_masks,
        final_mask,
    )

    # ------------------------------------------------------------------ #
    # 7. Contour detection and curve fitting
    # ------------------------------------------------------------------ #
    print("Fitting arc curves …")
    raw_arc_contours, _ = cv2.findContours(
        arc_edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    arc_contours = filter_arc_contours(list(raw_arc_contours), overarching_mask_w)

    match_result      = match_arcs_to_impressions(arc_contours, ordered_contours)
    closest_contours  = match_result["closest_4_Contours"]
    closest_distances = match_result["closest_contour_distances"]
    closest_centers   = match_result["closest_contour_centers"]
    orig_fitted       = match_result["originalFittedCurves"]

    if not orig_fitted:
        print("No arc curves matched to impressions — skipping measurement.")
        return

    arc_contours_1, lspace = build_ordered_arc_contours_1(
        orig_fitted, arc_edge_image.shape[1]
    )

    # Build overarching mask contour image (thickness=3 for intersection)
    oam_image = np.zeros(arc_edge_image.shape, np.uint8)
    cv2.drawContours(oam_image, [overarching_contour], 0, 255, thickness=3)

    arc2_result = build_ordered_arc_contours_2_and_intersections(
        ordered_centroids,
        arc_contours_1,
        ordered_masks,
        oam_image,
        overarching_mask_w,
        arc_edge_image.shape,
        lspace,
    )
    arc_contours_2       = arc2_result["orderedChosenArcContours_2"]
    intersection_pts     = arc2_result["intersectionPoints"]
    final_fitted         = arc2_result["finalFittedCurves"]
    to_delete            = arc2_result["impressionsToBeDeleted"]
    edge_contact_image   = arc2_result["edgeAndContactAndContour"]

    # ------------------------------------------------------------------ #
    # 8. Impression pruning
    # ------------------------------------------------------------------ #
    filter_impressions_by_intersection(
        to_delete,
        ordered_centroids, ordered_masks, ordered_contours, ordered_bounds,
        closest_contours, closest_distances, closest_centers,
        orig_fitted, arc_contours_1, arc_contours_2, final_fitted,
        intersection_pts,
    )

    filter_impressions_by_width(
        ordered_centroids, ordered_masks, ordered_contours, ordered_bounds,
        closest_contours, closest_distances, closest_centers,
        orig_fitted, arc_contours_1, arc_contours_2, final_fitted,
        intersection_pts,
    )

    filter_impressions_by_spacing(
        ordered_centroids, ordered_masks, ordered_contours, ordered_bounds,
        closest_contours, closest_distances, closest_centers,
        orig_fitted, arc_contours_1, arc_contours_2, final_fitted,
        intersection_pts,
    )

    if not arc_contours_2:
        print("No valid impression/arc pairs remain after filtering — skipping.")
        return

    # ------------------------------------------------------------------ #
    # 9. Measurements
    # ------------------------------------------------------------------ #
    print("Computing measurements …")
    image_for_display = copy.deepcopy(raw_image)

    meas = compute_measurements(
        arc_contours_2,
        ordered_contours,
        ordered_masks,
        intersection_pts,
        arc_len_mm,
        width_mult,
        overarching_mask,
        final_mask,
        image_for_display,
    )

    toe   = meas["toe_clearance_mm"]
    cl    = meas["contact_length_mm"]
    heel  = meas["heel_clearance_mm"]
    width = meas["contact_width_mm"]
    annotated_image = meas["image_for_final_display"]

    # ------------------------------------------------------------------ #
    # 10. Build dimensions table and composite output image
    # ------------------------------------------------------------------ #
    dim_table = make_dimensions_table(
        width=annotated_image.shape[1],
        height=DIMENSIONS_TABLE_HEIGHT,
        drive_coast_label="DRIVE SIDE",
        dimensions=(toe, cl, heel, width, 0),
        dimensions_tolerance=tolerances,
    )
    final_output = assemble_final_image(annotated_image, dim_table)

    # ------------------------------------------------------------------ #
    # 11. Display
    # ------------------------------------------------------------------ #
    render_results(
        enhanced_image=enhanced_yb_image,
        edge_contact_image=edge_contact_image,
        final_annotated_image=final_output,
        filename=filename,
    )

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"Completed {filename} in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the SAM model once, then process every image in the directory."""
    print("Loading SAM model …")
    sam = load_sam_components()

    image_files = list(iter_image_files(IMAGES_DIRECTORY))
    print(f"Found {len(image_files)} image(s) in '{IMAGES_DIRECTORY}'")

    for image_path in image_files:
        try:
            process_image(str(image_path), sam)
        except Exception as exc:
            print(f"ERROR processing {image_path.name}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()