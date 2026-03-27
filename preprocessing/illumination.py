"""
preprocessing/illumination.py
==============================
Implementation of the Ying 2017 CAIP low-light image enhancement
algorithm.

Reference
---------
Ying, Z., Li, G., Ren, Y., Wang, R., & Wang, W. (2017).
"A New Image Contrast Enhancement Algorithm Using Exposure Fusion
Framework." CAIP 2017.

The algorithm pipeline (as used in the notebook):
  1. ``computeTextureWeights``  — estimate structure/texture weights W_h, W_v
  2. ``solveLinearEquation``    — solve the sparse system to get a smoothed map S
  3. ``tsmooth``                — orchestrate steps 1+2 for a single channel
  4. ``rgb2gm``                 — geometric mean of RGB channels
  5. ``applyK``                 — apply camera exposure model with parameter k
  6. ``entropy``                — compute Shannon entropy of an image patch
  7. ``maxEntropyEnhance``      — find the optimal k that maximises entropy
  8. ``Ying_2017_CAIP``         — top-level entry point; returns enhanced uint8 image

Only ``Ying_2017_CAIP`` needs to be called by external code.
"""

import numpy as np
import cv2
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
import scipy.optimize
import skimage.transform


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def computeTextureWeights(fin: np.ndarray, sigma: float, sharpness: float):
    """
    Compute horizontal and vertical texture weights for the smoothing
    optimisation.

    Parameters
    ----------
    fin : np.ndarray
        Single-channel float image in [0, 1].
    sigma : float
        Gaussian window size for the local average (convolution width).
    sharpness : float
        Small regularisation constant to avoid division by zero.

    Returns
    -------
    W_h, W_v : np.ndarray, np.ndarray
        Horizontal and vertical weight matrices, same shape as *fin*.
    """
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0, :] - fin[-1, :]))
    dt0_h = np.vstack((
        np.diff(fin, n=1, axis=1).conj().T,
        fin[:, 0].conj().T - fin[:, -1].conj().T
    )).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1, sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma, 1)), mode='same')

    W_h = 1.0 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1.0 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)

    return W_h, W_v


def solveLinearEquation(IN: np.ndarray, wx: np.ndarray, wy: np.ndarray,
                        lamda: float) -> np.ndarray:
    """
    Solve the sparse linear system arising from the texture-smoothing
    optimisation:

        min_S  ||S - I||² + λ (W_h ||∂_h S||² + W_v ||∂_v S||²)

    Parameters
    ----------
    IN : np.ndarray
        Input single-channel float image.
    wx, wy : np.ndarray
        Horizontal and vertical texture weights (from ``computeTextureWeights``).
    lamda : float
        Regularisation strength (λ in the objective).

    Returns
    -------
    OUT : np.ndarray
        Smoothed image, same shape as *IN*.
    """
    r, c = IN.shape
    k = r * c

    dx  = -lamda * wx.flatten('F')
    dy  = -lamda * wy.flatten('F')
    dxa = -lamda * np.roll(wx, 1, axis=1).flatten('F')
    dya = -lamda * np.roll(wy, 1, axis=0).flatten('F')

    # boundary handling
    tmp = wx[:, -1]
    tempx = np.concatenate((tmp[:, None], np.zeros((r, c - 1))), axis=1)
    tmp = wy[-1, :]
    tempy = np.concatenate((tmp[None, :], np.zeros((r - 1, c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')

    wx[:, -1] = 0
    wy[-1, :] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')

    Ax = scipy.sparse.spdiags(
        np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T,
        np.array([-k + r, -r]), k, k
    )
    Ay = scipy.sparse.spdiags(
        np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0),
        np.array([-r + 1, -1]), k, k
    )
    D = 1 - (dx + dy + dxa + dya)
    A = ((Ax + Ay) + (Ax + Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T

    tin = IN[:, :]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')
    return OUT


def tsmooth(img: np.ndarray, lamda: float = 0.01, sigma: float = 3.0,
            sharpness: float = 0.001) -> np.ndarray:
    """
    Apply texture-aware smoothing to a single-channel image.

    Parameters
    ----------
    img : np.ndarray
        Single-channel input image (any dtype; will be normalised internally).
    lamda : float
        Regularisation strength (default 0.01).
    sigma : float
        Gaussian window size for texture weight estimation (default 3.0).
    sharpness : float
        Small regularisation constant (default 0.001).

    Returns
    -------
    np.ndarray
        Smoothed image in [0, 1] float64.
    """
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S


def rgb2gm(I: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to its per-pixel geometric mean (single channel).

    Parameters
    ----------
    I : np.ndarray
        RGB float image in [0, 1], shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Geometric mean image, shape (H, W), values in [0, 1].
    """
    if I.shape[2] == 3:
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs(I[:, :, 0] * I[:, :, 1] * I[:, :, 2]) ** (1.0 / 3.0)
    return I


def applyK(I: np.ndarray, k: float, a: float = -0.3293,
           b: float = 1.1258) -> np.ndarray:
    """
    Apply the camera exposure model with parameter *k*.

    f(k) = exp((1 - k^a) * b)   [β]
    γ    = k^a
    J    = I^γ * β

    Parameters
    ----------
    I : np.ndarray
        Input image (float, arbitrary range).
    k : float
        Exposure ratio.
    a, b : float
        Camera model constants.

    Returns
    -------
    np.ndarray
        Exposure-corrected image, same shape and dtype as *I*.
    """
    beta  = np.exp((1.0 - k ** a) * b)
    gamma = k ** a
    return (I ** gamma) * beta


def entropy(X: np.ndarray) -> float:
    """
    Compute the Shannon entropy (base-2) of image *X*.

    The image is clipped to [0, 255] and treated as uint8 before
    computing the histogram.

    Parameters
    ----------
    X : np.ndarray
        Input image or patch (float, values nominally in [0, 1] scaled ×255
        before histogram).

    Returns
    -------
    float
        Entropy in bits.
    """
    tmp = X * 255
    tmp = np.clip(tmp, 0, 255).astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = counts.astype(float) / counts.sum()
    return float(-np.sum(pk * np.log2(pk)))


def maxEntropyEnhance(I: np.ndarray, isBad: np.ndarray,
                      a: float = -0.3293, b: float = 1.1258) -> np.ndarray:
    """
    Find the exposure ratio *k* ∈ [1, 7] that maximises the entropy of the
    under-exposed pixels, then apply the resulting exposure correction to
    the full image *I*.

    Parameters
    ----------
    I : np.ndarray
        Full RGB float image in [0, 1], shape (H, W, 3).
    isBad : np.ndarray
        Boolean mask (H, W) — True where pixels are considered under-exposed.
    a, b : float
        Camera model constants.

    Returns
    -------
    np.ndarray
        Exposure-corrected RGB image, same shape as *I*.
    """
    isBad = isBad.astype(float)
    size_i = np.array(isBad.shape)[0:2]
    new_size_i = tuple((size_i * 0.5).astype(int))

    tmp = skimage.transform.resize(I, new_size_i)
    tmp = np.clip(tmp.real, 0, None)
    Y = rgb2gm(tmp)

    isBad_small = skimage.transform.resize(isBad, new_size_i, order=3)
    isBad_small = (isBad_small >= 0.5).astype(float)
    Y = Y[isBad_small == 1]

    if Y.size == 0:
        return I

    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)

    J = applyK(I, opt_k, a, b) - 0.01
    return J


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def Ying_2017_CAIP(img: np.ndarray, mu: float = 0.5,
                   a: float = -0.3293, b: float = 1.1258) -> np.ndarray:
    """
    Enhance a low-light RGB image using the Ying 2017 CAIP method.

    The algorithm fuses an original exposure estimate *I* with an
    entropy-maximised exposure estimate *J* using a weight matrix derived
    from the smoothed luminance map.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image (uint8 or float), shape (H, W, 3).
    mu : float
        Fusion weight exponent (default 0.5).
    a, b : float
        Camera model constants.

    Returns
    -------
    np.ndarray
        Enhanced RGB image, shape (H, W, 3), dtype uint8, values in [0, 255].
    """
    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Luminance map: take max across channels
    t_b = np.max(I, axis=2)

    size_i = np.array(t_b.shape)[0:2]
    new_size_i = tuple((size_i * 0.5).astype(int))
    resized_image = skimage.transform.resize(t_b, new_size_i, order=3)

    t_our = cv2.resize(
        tsmooth(resized_image, lamda, sigma),
        (t_b.shape[1], t_b.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)

    # Build 3-channel weight matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:, :, i] = t_our
    W = t ** mu

    result = I * W + J * (1.0 - W)
    result = np.clip(result * 255, 0, 255)
    return result.astype(np.uint8)