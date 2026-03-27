"""
config/settings.py
==================
Central configuration for the gear inspection pipeline.

Contains all hardcoded parameters that were scattered across the original
notebook:  directory paths, SAM model settings, per-component calibration
data (arc lengths, tolerances, width multipliers), SAM prompt points, and
colour/display constants.

To add a new component type, add an entry to each of the four
dictionaries (theInputPoints, arcLengthsDict, arcLengthsToleranceDict,
widthMultiplierDict) keyed by the relevant substring that appears in the
image filename.
"""

import os

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# Base directory for all Dana inspection data.
# Override by setting the DANA_DIR environment variable, or edit directly.
DANA_DIRECTORY: str = os.environ.get(
    "DANA_DIR",
    r"C:\Dana"          # <-- change to your local path
)

IMAGES_DIRECTORY:          str = os.path.join(DANA_DIRECTORY, "images")
MODEL_WEIGHTS_DIRECTORY:   str = os.path.join(DANA_DIRECTORY, "model_checkpoints", "weights")
FACEBOOK_MODELS_DIRECTORY: str = os.path.join(DANA_DIRECTORY, "model_checkpoints", "facebook_models")

# ---------------------------------------------------------------------------
# SAM model settings
# ---------------------------------------------------------------------------
# Filename of the SAM checkpoint inside MODEL_WEIGHTS_DIRECTORY.
# Supported variants: sam_vit_h_4b8939.pth  /  sam_vit_l_0b3195.pth  /  sam_vit_b_01ec64.pth
SAM_CHECKPOINT_FILENAME: str = "sam_vit_l_0b3195.pth"

# Sub-folder inside FACEBOOK_MODELS_DIRECTORY that holds the HuggingFace
# SamModel / SamProcessor files (e.g. "large", "base", "huge").
FACEBOOK_MODEL_SUBFOLDER: str = "large"

# ---------------------------------------------------------------------------
# SAM prompt points
# ---------------------------------------------------------------------------
# Keyed by a substring that uniquely identifies the component in the filename.
# Each value is a list of point-sets in the format expected by SamProcessor:
#   [ [ [x1,y1], [x2,y2], ... ] ]   (one list of points → one mask output)
#
# All coordinates are for the ORIGINAL (full-resolution) image.
# "Try 3" configuration from the notebook.
theInputPoints: dict = {}

theInputPoints["DIC01230119"] = [[[260, 360], [200, 280], [140, 200]]]
theInputPoints["DIC02230424"] = [[[220, 320], [180, 245], [140, 170]]]
theInputPoints["DIC02230425"] = [[[220, 320], [165, 245], [110, 170]]]
theInputPoints["DIC02230609"] = [[[280, 360], [210, 245], [140, 130]]]
theInputPoints["DIC03230118"] = [[[430, 340], [390, 240], [350, 140]]]
theInputPoints["______"]      = [[[380, 410], [360, 280], [340, 150]]]
theInputPoints["DIC01230202P"] = [[[230, 400], [175, 280], [120, 160]]]
theInputPoints["DIC01230224P"] = [[[260, 360], [210, 260], [160, 160]]]
theInputPoints["DIC01230313P"] = [[[280, 400], [220, 260], [160, 120]]]
theInputPoints["DIC01230531P"] = [[[220, 400], [170, 270], [120, 140]]]
theInputPoints["DIC02230321P"] = [[[200, 420], [160, 270], [120, 120]]]
theInputPoints["DIC02230509P"] = [[[220, 400], [180, 270], [140, 140]]]

# ---------------------------------------------------------------------------
# Per-component calibration data
# ---------------------------------------------------------------------------
# Known physical arc length (mm) across the full gear tooth for each
# component family.  Used to scale pixel arc-length measurements to mm.
# Default (fallback) = 64.0 mm.
arcLengthsDict: dict = {
    "DIC03230118": 59.9,
    "DIC02230609": 68.12,
    "DIC01230224": 66.60,
    "DIC01230202": 66.64,
    "DIC01230119": 66.60,
}

# Acceptable [min, max] measurement ranges (mm) for each of the five metrics:
#   index 0 → Toe Clearance
#   index 1 → Contact Length
#   index 2 → Heel Clearance
#   index 3 → Contact Width
#   index 4 → Tip Clearance
# Default (fallback) tolerances are for the general P-series components.
arcLengthsToleranceDict: dict = {
    "DIC03230118": [[3, 10], [31, 53.9], [3, 23],  [8, 11], [1, 4]],
    "DIC02230609": [[3, 10], [29, 47],   [10, 28], [8, 11], [1, 4]],
    "DIC01230224": [[3, 10], [29, 47],   [10, 26], [8, 11], [1, 4]],
    "DIC01230202": [[3, 10], [29, 47],   [10, 26], [8, 11], [1, 4]],
    "DIC01230119": [[3, 10], [29, 47],   [10, 26], [8, 11], [1, 4]],
}
DEFAULT_TOLERANCES: list = [[3, 10], [29, 47], [10, 36], [8, 11], [1, 4]]

# Multiplier applied to the raw pixel width measurement before converting to
# mm.  Accounts for viewing angle / perspective differences per component.
# Default (fallback) = 1.35.
widthMultiplierDict: dict = {
    "DIC03230118": 1.35,
    "DIC02230609": 1.40,
    "DIC01230224": 1.40,
    "DIC01230202": 1.40,
    "DIC01230119": 1.40,
}
DEFAULT_WIDTH_MULTIPLIER: float = 1.35
DEFAULT_ARC_LENGTH_MM:    float = 64.0

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
# The canonical "yellow" pixel colour used for enhancement operations.
THE_YELLOW_PIXEL: tuple = (225, 225, 125)

# Dimensions table layout
DIMENSIONS_TABLE_HEIGHT: int = 250   # pixels
DIMENSIONS_TABLE_WIDTH_DEFAULT: int = 640