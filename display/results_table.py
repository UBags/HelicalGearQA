"""
display/results_table.py
=========================
Render the measurement results table and alpha-overlay helper.

Functions
---------
make_dimensions_table
    Build a white OpenCV image containing the five-row measurement table
    with green/red pass-fail colouring.

overlay_image_alpha
    Blend an RGBA overlay onto an RGB base image at a given position.

assemble_final_image
    Stack the annotated photo and the dimensions table vertically.
"""

from __future__ import annotations

import numpy as np
import cv2

from config.settings import DIMENSIONS_TABLE_HEIGHT


# ---------------------------------------------------------------------------
# Alpha overlay
# ---------------------------------------------------------------------------

def overlay_image_alpha(img: np.ndarray,
                          img_overlay: np.ndarray,
                          x: int, y: int,
                          alpha_mask: np.ndarray) -> None:
    """
    Blend *img_overlay* onto *img* at pixel position *(x, y)* using
    *alpha_mask* for per-pixel opacity.  Modifies *img* in place.

    Parameters
    ----------
    img : np.ndarray
        Base RGB image (H, W, 3), dtype uint8.
    img_overlay : np.ndarray
        Overlay RGB image (h, w, 3), dtype uint8.
    x, y : int
        Top-left corner coordinates on *img* where the overlay is placed.
    alpha_mask : np.ndarray
        Float array (h, w) with values in [0, 1].  1 = fully overlay;
        0 = fully base.
    """
    y1 = max(0, y);          y2 = min(img.shape[0], y + img_overlay.shape[0])
    x1 = max(0, x);          x2 = min(img.shape[1], x + img_overlay.shape[1])
    y1o = max(0, -y);         y2o = min(img_overlay.shape[0], img.shape[0] - y)
    x1o = max(0, -x);         x2o = min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha     = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    img[y1:y2, x1:x2] = (
        alpha     * img_overlay[y1o:y2o, x1o:x2o] +
        alpha_inv * img[y1:y2, x1:x2]
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dimensions table
# ---------------------------------------------------------------------------

def make_dimensions_table(width:              int   = 640,
                            height:             int   = DIMENSIONS_TABLE_HEIGHT,
                            drive_coast_label:  str   = "DRIVE SIDE",
                            dimensions:         tuple = (0, 0, 0, 0, 0),
                            dimensions_tolerance: list = None) -> np.ndarray:
    """
    Render a white OpenCV image containing the measurement table.

    Layout (7 rows):
      Row 0  — "DRIVE SIDE" header
      Row 1  — "DESCRIPTION" / "MEASUREMENT [OK RANGE]" column headers
      Row 2  — TOE CLEARANCE
      Row 3  — CONTACT LENGTH
      Row 4  — HEEL CLEARANCE
      Row 5  — CONTACT WIDTH
      Row 6  — TIP CLEARANCE

    Dimension values that fall within their tolerance range are rendered
    in green; out-of-range values are rendered in red.

    Parameters
    ----------
    width : int
        Table width in pixels (default 640).
    height : int
        Table height in pixels (default ``DIMENSIONS_TABLE_HEIGHT`` = 250).
    drive_coast_label : str
        Header label, e.g. "DRIVE SIDE" or "COAST SIDE".
    dimensions : tuple of 5 floats
        (toe_clearance, contact_length, heel_clearance,
         contact_width, tip_clearance)  in mm.
    dimensions_tolerance : list of [float, float] or None
        Five [min, max] pairs (mm).  If None, a permissive default is
        used (all values pass).

    Returns
    -------
    np.ndarray
        RGB image of the table, shape (height, width, 3), dtype uint8.
    """
    if dimensions_tolerance is None:
        dimensions_tolerance = [[0, 9999]] * 5

    y_gap   = 5
    x_gap   = 5
    font    = cv2.FONT_HERSHEY_COMPLEX
    scale   = 0.65
    thick   = 2
    blue    = (0, 0, 255)
    black   = (0, 0, 0)
    green   = (80, 180, 50)
    red     = (230, 50, 20)
    gap     = (height - 2 * y_gap) // 7

    def _centre_x(text: str, col_centre: int) -> int:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        return col_centre - tw // 2

    def _row_y(row: int) -> int:
        (_, th), _ = cv2.getTextSize("A", font, scale, thick)
        return y_gap + row * gap + th + (gap - th) // 2

    left_cx  = width // 4
    right_cx = width - width // 4
    mid_x    = width // 2

    img = np.ones([height, width, 3], dtype=np.uint8) * 255

    # --- Grid lines ---
    for pt1, pt2, lw in [
        ((x_gap, y_gap),           (width - x_gap, y_gap),            2),
        ((x_gap, height - y_gap),  (width - x_gap, height - y_gap),   2),
        ((x_gap, y_gap),           (x_gap, height - y_gap),           2),
        ((width - x_gap, y_gap),   (width - x_gap, height - y_gap),   2),
        ((x_gap, y_gap + gap),     (width - x_gap, y_gap + gap),      1),
        ((x_gap, y_gap + 2*gap),   (width - x_gap, y_gap + 2*gap),   1),
        ((x_gap, y_gap + 3*gap),   (width - x_gap, y_gap + 3*gap),   1),
        ((x_gap, height - gap),    (width - x_gap, height - gap),     1),
        ((x_gap, height - 2*gap),  (width - x_gap, height - 2*gap),  1),
        ((x_gap, height - 3*gap),  (width - x_gap, height - 3*gap),  1),
        ((mid_x, gap),             (mid_x, height),                   1),
    ]:
        cv2.line(img, pt1, pt2, black, lw)

    # --- Static labels ---
    labels = [
        ("DRIVE SIDE",           _centre_x("DRIVE SIDE",          left_cx),    _row_y(0), black),
        ("DESCRIPTION",          _centre_x("DESCRIPTION",         left_cx),    _row_y(1), black),
        ("MEASUREMENT [OK RANGE]",_centre_x("MEASUREMENT [OK RANGE]", right_cx), _row_y(1), black),
        ("TOE CLEARANCE",        _centre_x("TOE CLEARANCE",        left_cx),   _row_y(2), blue),
        ("CONTACT LENGTH",       _centre_x("CONTACT LENGTH",       left_cx),   _row_y(3), blue),
        ("HEEL CLEARANCE",       _centre_x("HEEL CLEARANCE",       left_cx),   _row_y(4), blue),
        ("CONTACT WIDTH",        _centre_x("CONTACT WIDTH",        left_cx),   _row_y(5), blue),
        ("TIP CLEARANCE",        _centre_x("TIP CLEARANCE",        left_cx),   _row_y(6), blue),
    ]
    # Override header label
    labels[0] = (drive_coast_label,
                 _centre_x(drive_coast_label, left_cx),
                 _row_y(0), black)

    for text, x, y, colour in labels:
        cv2.putText(img, text, (x, y), font, scale, colour, thick, 2)

    # --- Measurement values ---
    row_offsets = [2, 3, 4, 5, 6]
    for idx, (dim, tol, row) in enumerate(
            zip(dimensions, dimensions_tolerance, row_offsets)):
        value_str = f"{dim} mm {str(tol).replace(', ', '-')}"
        x_val     = _centre_x(value_str, right_cx)
        y_val     = _row_y(row)
        in_range  = (tol[0] <= dim <= tol[1])
        colour    = green if in_range else red
        cv2.putText(img, value_str, (x_val, y_val), font, scale, colour, thick, 2)

    return img


# ---------------------------------------------------------------------------
# Final image assembly
# ---------------------------------------------------------------------------

def assemble_final_image(image_for_final_display: np.ndarray,
                           dimensions_table: np.ndarray) -> np.ndarray:
    """
    Stack *image_for_final_display* and *dimensions_table* vertically
    on a white canvas.

    Parameters
    ----------
    image_for_final_display : np.ndarray
        Annotated photo (H_photo, W, 3), uint8.
    dimensions_table : np.ndarray
        Table image (H_table, W, 3), uint8 — must have the same width.

    Returns
    -------
    np.ndarray
        Combined image (H_photo + H_table, W, 3), uint8.
    """
    h_photo, w_photo = image_for_final_display.shape[:2]
    h_table          = dimensions_table.shape[0]

    canvas = np.ones((h_photo + h_table, w_photo, 3), dtype=np.uint8) * 255
    canvas[0:h_photo,        0:w_photo] = image_for_final_display
    canvas[h_photo:h_photo + h_table, 0:w_photo] = dimensions_table
    return canvas