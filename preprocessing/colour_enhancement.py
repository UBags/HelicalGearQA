"""
preprocessing/colour_enhancement.py
=====================================
Colour, contrast, gamma, and channel-specific enhancement helpers.

Functions
---------
determine_gamma              — estimate a suitable gamma correction value
apply_gamma                  — apply gamma correction via a LUT
get_rgbcolor_equalized_image — CLAHE equalisation while preserving colour ratio
enhance_yellows_and_blacks   — row-wise yellow/black pixel classification and
                               selective replacement
apply_brightness_contrast    — simple brightness/contrast adjustment
change_contrast_brightness   — CLAHE on the L channel (LAB colour space)
enhance_rg_suppress_b        — CLAHE on the a/b channels (boost RG, dampen B)

All functions expect and return RGB numpy arrays (dtype uint8) unless
documented otherwise.
"""

import math
import copy
import functools

import numpy as np
import cv2
from scipy.signal import convolve2d

from config.settings import THE_YELLOW_PIXEL


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------

def determine_gamma(rgbimage: np.ndarray) -> float:
    """
    Estimate a gamma correction value from the dark pixels of *rgbimage*.

    Pixels with Value (HSV) ≥ 99 are excluded so that already-bright
    regions do not skew the estimate.  The result is boosted for
    mid-range gamma values to avoid over-darkening.

    Parameters
    ----------
    rgbimage : np.ndarray
        RGB image, shape (H, W, 3), dtype uint8.

    Returns
    -------
    float
        Recommended gamma value (> 1 → darken, < 1 → brighten).
    """
    hsv = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2HSV)
    _, _, val = cv2.split(hsv)

    mid = 0.5
    val_dark = val[val < 99]
    mean = np.mean(val_dark)
    gamma = math.log(mid * 255) / math.log(mean)

    if gamma < 1:
        return gamma
    if gamma < 1.25:
        return (gamma - 1) * 3 + 1
    if gamma < 1.5:
        return (gamma - 1) * 2 + 1
    return gamma


def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to *image* using a 256-entry look-up table.

    Parameters
    ----------
    image : np.ndarray
        RGB image, dtype uint8.
    gamma : float
        Gamma value.  Values < 1 brighten; values > 1 darken.

    Returns
    -------
    np.ndarray
        Gamma-corrected image, same shape and dtype as *image*.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
    ).astype(np.uint8)
    return cv2.LUT(image, table)


# ---------------------------------------------------------------------------
# Histogram equalisation
# ---------------------------------------------------------------------------

def get_rgbcolor_equalized_image(image: np.ndarray,
                                 clipLimit: float = 8.0,
                                 tileGridSize: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel while preserving colour ratios.

    Each colour channel is scaled by the ratio of the CLAHE-equalised
    grayscale value to the original grayscale value, which boosts
    contrast without shifting hue.

    Parameters
    ----------
    image : np.ndarray
        RGB image, dtype uint8.
    clipLimit : float
        CLAHE clip limit (default 8.0).
    tileGridSize : tuple of (int, int)
        CLAHE tile grid size (default (8, 8)).

    Returns
    -------
    np.ndarray
        Colour-equalised RGB image, dtype uint8.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    image_gray_eq = clahe.apply(image_gray)

    # avoid division by zero
    image_gray_safe = image_gray.astype(float)
    image_gray_safe[image_gray_safe == 0] = 1
    pixel_ratio = image_gray_eq / image_gray_safe

    r_ch, g_ch, b_ch = cv2.split(image)
    r_ch = np.clip(r_ch.astype(np.int32) * pixel_ratio, 0, 255).astype(np.uint8)
    g_ch = np.clip(g_ch.astype(np.int32) * pixel_ratio, 0, 255).astype(np.uint8)
    b_ch = np.clip(b_ch.astype(np.int32) * pixel_ratio, 0, 255).astype(np.uint8)

    image_eq = cv2.merge([b_ch, g_ch, r_ch])   # merge as BGR …
    image_eq = cv2.cvtColor(image_eq, cv2.COLOR_BGR2RGB)   # … then convert to RGB
    return image_eq


# ---------------------------------------------------------------------------
# Yellow / Black selective enhancement
# ---------------------------------------------------------------------------

def enhance_yellows_and_blacks(
        image: np.ndarray,
        kernelSize: int = 5,
        percentileCutoff: float = 12.5,
        grayRatioCutoff: float = 1.0,
        toTranspose: bool = True,
        changeBlacks: bool = True,
        changeYellows: bool = False,
        rgCutoff: tuple = (70, 70),
        rgbDiff: int = 30) -> np.ndarray:
    """
    Classify pixels row-by-row as 'yellow', 'black', or 'other' and
    optionally replace them with canonical colours.

    The function first optionally transposes the image so that the
    per-row neighbourhood statistics are computed along the longer axis.
    It uses a local-average kernel to estimate the neighbourhood
    brightness, then applies two-tier yellow detection and black
    detection rules.

    Parameters
    ----------
    image : np.ndarray
        RGB image, dtype uint8.
    kernelSize : int
        Neighbourhood kernel size for local-average computation (odd, ≥ 3).
    percentileCutoff : float
        Lower percentile of the local brightness distribution below which
        a pixel may be classified as black (default 12.5).
    grayRatioCutoff : float
        Maximum allowed ratio of local-average to actual grayscale for a
        pixel to be classified as black (default 1.0).
    toTranspose : bool
        If True, transpose the image before processing and un-transpose
        the result (default True — operates along the wider axis).
    changeBlacks : bool
        If True, mark black pixels as (15, 15, 15) (default True).
    changeYellows : bool
        If True, mark yellow pixels as THE_YELLOW_PIXEL (default False).
    rgCutoff : tuple of (int, int)
        Minimum R and G channel values for the strict yellow criterion
        (default (70, 70)).
    rgbDiff : int
        Minimum difference between R/G and B channels required for the
        strict yellow criterion (default 30).

    Returns
    -------
    np.ndarray
        Modified image, same shape and dtype as *image*.
    """
    if toTranspose:
        img_work = cv2.transpose(image)
    else:
        img_work = image.copy()

    # sanitise kernel size
    kernelSize = max(kernelSize, 3)
    if kernelSize % 2 == 0:
        kernelSize += 1

    r_ch, g_ch, b_ch = cv2.split(img_work)
    r_ch = r_ch.astype(np.int32)
    g_ch = g_ch.astype(np.int32)
    b_ch = b_ch.astype(np.int32)

    kernel = np.ones((kernelSize, kernelSize), dtype=np.float32)
    kernel /= kernel.sum()
    r_avg = np.clip(convolve2d(r_ch, kernel, boundary='symm', mode='same'), 0, 255).astype(np.uint8)
    g_avg = np.clip(convolve2d(g_ch, kernel, boundary='symm', mode='same'), 0, 255).astype(np.uint8)
    b_avg = np.clip(convolve2d(b_ch, kernel, boundary='symm', mode='same'), 0, 255).astype(np.uint8)

    avg_bgr = cv2.merge([b_avg, g_avg, r_avg])
    avg_rgb = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2RGB)
    transposed_gray      = cv2.cvtColor(img_work, cv2.COLOR_RGB2GRAY)
    transposed_avg_gray  = cv2.cvtColor(avg_rgb,  cv2.COLOR_RGB2GRAY)

    transposed_gray_safe = transposed_gray.astype(float)
    transposed_gray_safe[transposed_gray_safe == 0] = 1
    gray_ratio = transposed_avg_gray.astype(float) / transposed_gray_safe

    for i in range(img_work.shape[0]):
        actual_pixels      = img_work[i]
        average_gray_row   = transposed_avg_gray[i]
        g_ratio_row        = gray_ratio[i]

        non_bright = average_gray_row[average_gray_row < 253]
        percentile_cut = np.percentile(non_bright, percentileCutoff) \
            if (non_bright is not None and non_bright.shape[0] > 5) else 0

        # Strict yellow: high R, G, both much higher than B
        arr_y1 = [
            r_ch[i] > rgCutoff[0], g_ch[i] > rgCutoff[1],
            (r_ch[i] - b_ch[i]) > rgbDiff,
            (g_ch[i] - b_ch[i]) > rgbDiff,
            b_ch[i] < 170
        ]
        possibly_yellow1 = functools.reduce(np.logical_and, arr_y1)

        # Loose yellow: lower thresholds
        arr_y2 = [
            r_ch[i] > 50, g_ch[i] > 54,
            (r_ch[i] - b_ch[i]) > 8,
            (g_ch[i] - b_ch[i]) > 8,
            b_ch[i] < 170
        ]
        possibly_yellow2 = functools.reduce(np.logical_and, arr_y2)
        possibly_yellow  = np.logical_or(possibly_yellow1, possibly_yellow2)

        # Black detection
        arr_black = [
            g_ratio_row < grayRatioCutoff,
            average_gray_row < percentile_cut,
            average_gray_row < 100,
            (r_ch[i] - b_ch[i]) < rgbDiff,
            (g_ch[i] - b_ch[i]) < rgbDiff,
        ]
        possibly_black = functools.reduce(np.logical_and, arr_black)

        # Definite black override
        def_black = functools.reduce(np.logical_and,
                                     [r_ch[i] < 41, g_ch[i] < 43, b_ch[i] < 43])
        possibly_black = np.logical_or(possibly_black, def_black)

        if changeBlacks:
            mark_black = np.logical_and(possibly_black, ~possibly_yellow)
            actual_pixels[mark_black] = (15, 15, 15)
        if changeYellows:
            mark_yellow = np.logical_and(~possibly_black, possibly_yellow)
            actual_pixels[mark_yellow] = THE_YELLOW_PIXEL

    if toTranspose:
        return cv2.transpose(img_work)
    return img_work.copy()


# ---------------------------------------------------------------------------
# Brightness / contrast
# ---------------------------------------------------------------------------

def apply_brightness_contrast(input_img: np.ndarray,
                               brightness: int = 0,
                               contrast: int = 0) -> np.ndarray:
    """
    Adjust brightness and contrast of *input_img* using linear blending.

    Parameters
    ----------
    input_img : np.ndarray
        RGB image, dtype uint8.
    brightness : int
        Brightness offset in the range (-255, 255).  Positive → brighter.
    contrast : int
        Contrast adjustment in the range (-127, 127).  Positive → more contrast.

    Returns
    -------
    np.ndarray
        Adjusted image, same shape and dtype as *input_img*.
    """
    if brightness != 0:
        shadow    = brightness if brightness > 0 else 0
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b   = (highlight - shadow) / 255.0
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, float(shadow))
    else:
        buf = input_img.copy()

    if contrast != 0:
        f       = 131.0 * (contrast + 127) / (127.0 * (131 - contrast))
        gamma_c = 127.0 * (1.0 - f)
        buf = cv2.addWeighted(buf, f, buf, 0, gamma_c)

    return buf


def change_contrast_brightness(rgbimg: np.ndarray,
                                clipLimit: float = 2.0) -> np.ndarray:
    """
    Increase contrast by applying CLAHE to the L channel in LAB colour space.

    Parameters
    ----------
    rgbimg : np.ndarray
        RGB image, dtype uint8.
    clipLimit : float
        CLAHE clip limit (default 2.0).

    Returns
    -------
    np.ndarray
        Contrast-enhanced RGB image, dtype uint8.
    """
    lab = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    new_l = clahe.apply(l_channel)
    limg = cv2.merge((new_l, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def enhance_rg_suppress_b(rgbimg: np.ndarray,
                           clipLimit: float = 2.0) -> np.ndarray:
    """
    Boost the red/green channels and subtly suppress blue by applying
    CLAHE to the a and b channels of the LAB colour space.

    Parameters
    ----------
    rgbimg : np.ndarray
        RGB image, dtype uint8.
    clipLimit : float
        CLAHE clip limit for the a-channel (default 2.0).
        The b-channel uses a fixed lower clip limit (0.75).

    Returns
    -------
    np.ndarray
        Channel-enhanced RGB image, dtype uint8.
    """
    lab = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe_a = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    new_a   = clahe_a.apply(a)

    clahe_b = cv2.createCLAHE(clipLimit=0.75, tileGridSize=(8, 8))
    new_b   = clahe_b.apply(b)

    limg = cv2.merge((l_channel, new_a, new_b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)