"""
preprocessing/segmentation_preprocess.py
==========================================
Colour-quantisation helpers used to reduce the gear tooth image to a
small number of representative colour clusters before arc-edge detection.

Two backends are provided:
  - OpenCV K-Means  (primary, used in the pipeline)
  - sklearn MiniBatchKMeans  (alternative)
  - PIL quantise  (alternative)

The main entry point used by the pipeline is ``get_kmeans_segmented_image``.

All functions expect RGB numpy arrays and return RGB numpy arrays unless
documented otherwise.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# OpenCV K-Means (primary pipeline)
# ---------------------------------------------------------------------------

def kmeans_preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Reshape *image* to a 2-D array of pixels and convert to float32,
    ready for ``cv2.kmeans``.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Float32 array, shape (H*W, 3).
    """
    return image.reshape((-1, 3)).astype(np.float32)


def perform_kmeans_clustering(pixel_values: np.ndarray,
                               k: int = 3) -> tuple:
    """
    Run OpenCV K-Means clustering on a (N, 3) float32 array.

    Parameters
    ----------
    pixel_values : np.ndarray
        Float32 array of pixel values, shape (N, 3).
    k : int
        Number of clusters (default 3).

    Returns
    -------
    compactness : float
        Sum of squared distances to the nearest cluster centre.
    labels : np.ndarray
        Integer label array, shape (N, 1).
    centers : np.ndarray
        Cluster centroids, shape (k, 3), dtype uint8.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return compactness, labels, np.uint8(centers)


def create_kmeans_segmented_image(originalImage: np.ndarray,
                                  pixel_values: np.ndarray,
                                  labels: np.ndarray,
                                  centers: np.ndarray) -> np.ndarray:
    """
    Reconstruct a segmented image by mapping each pixel to its cluster centre.

    Parameters
    ----------
    originalImage : np.ndarray
        Original image (used only to infer the output shape).
    pixel_values : np.ndarray
        Float32 pixel array (unused; kept for API consistency).
    labels : np.ndarray
        Cluster label per pixel, shape (N, 1).
    centers : np.ndarray
        Cluster centres, shape (k, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        Segmented image, same shape as *originalImage*, dtype uint8.
    """
    segmented = centers[labels.flatten()]
    return segmented.reshape(originalImage.shape)


def create_kmeans_masked_image(originalImage: np.ndarray,
                               labels: np.ndarray,
                               cluster_to_disable: int) -> np.ndarray:
    """
    Return a copy of *originalImage* with all pixels belonging to
    *cluster_to_disable* set to black (0, 0, 0).

    Parameters
    ----------
    originalImage : np.ndarray
        Original RGB image.
    labels : np.ndarray
        Cluster label per pixel, shape (N, 1).
    cluster_to_disable : int
        Index of the cluster to suppress.

    Returns
    -------
    np.ndarray
        Masked image, same shape as *originalImage*.
    """
    masked = np.copy(originalImage).reshape((-1, 3))
    masked[labels.flatten() == cluster_to_disable] = [0, 0, 0]
    return masked.reshape(originalImage.shape)


def get_kmeans_segmented_image(image: np.ndarray,
                                no_of_segments: int = 3,
                                resize: bool = False,
                                mask_cluster: int | None = None) -> tuple:
    """
    Apply K-Means colour quantisation to *image* and return the result.

    Parameters
    ----------
    image : np.ndarray
        RGB image, dtype uint8.
    no_of_segments : int
        Number of colour clusters (default 3).
    resize : bool
        If True, process a half-resolution copy and up-scale the result;
        this speeds up clustering on large images (default False).
    mask_cluster : int or None
        If given, zero out (black) the cluster at this index rather than
        producing a fully segmented image.  If the index is ≥ no_of_segments,
        the full segmented image is returned instead (default None).

    Returns
    -------
    segmented_image : np.ndarray
        Quantised (or masked) RGB image, same spatial resolution as *image*.
    centers : np.ndarray
        Cluster centre colours, shape (no_of_segments, 3), dtype uint8.
    """
    if resize:
        current_size = np.array(image.shape)[0:2]
        new_size     = tuple((current_size * 0.5).astype(int))
        work_image   = cv2.resize(image, (new_size[1], new_size[0]))
    else:
        work_image = image

    pixel_values                = kmeans_preprocess_image(work_image)
    compactness, labels, centers = perform_kmeans_clustering(pixel_values, no_of_segments)

    if mask_cluster is None:
        segmented = create_kmeans_segmented_image(work_image, pixel_values, labels, centers)
    elif mask_cluster < no_of_segments:
        segmented = create_kmeans_masked_image(work_image, labels, mask_cluster)
    else:
        segmented = create_kmeans_segmented_image(work_image, pixel_values, labels, centers)

    if resize:
        final_image = cv2.resize(segmented, (current_size[1], current_size[0]))
    else:
        final_image = segmented

    return final_image, centers


# ---------------------------------------------------------------------------
# Sklearn backend (alternative — not used in the primary pipeline)
# ---------------------------------------------------------------------------

def color_quantize_sklearn(rgbImage: np.ndarray,
                            no_of_clusters: int = 3) -> tuple:
    """
    Colour quantise *rgbImage* using sklearn's MiniBatchKMeans.

    Operates on a half-resolution copy for speed; the result is
    upscaled back to the original resolution.

    Parameters
    ----------
    rgbImage : np.ndarray
        RGB image, dtype uint8.
    no_of_clusters : int
        Number of colour clusters (default 3).

    Returns
    -------
    finalImage : np.ndarray
        Quantised RGB image at original resolution.
    centers : np.ndarray
        Cluster centres (uint8).
    """
    current_size = np.array(rgbImage.shape)[0:2]
    new_size     = tuple((current_size * 0.5).astype(int))
    resized      = cv2.resize(rgbImage, (new_size[1], new_size[0]))

    km = MiniBatchKMeans(no_of_clusters, compute_labels=False)
    flat = resized.reshape(-1, 1)
    km.fit(flat)
    labels  = km.predict(flat)
    centers = np.uint8(km.cluster_centers_)
    q_img   = np.uint8(km.cluster_centers_[labels].reshape(resized.shape))
    final   = cv2.resize(q_img, (current_size[1], current_size[0]))
    return final, centers


# ---------------------------------------------------------------------------
# PIL backend (alternative — not used in the primary pipeline)
# ---------------------------------------------------------------------------

def color_quantize_PIL(rgbImage: np.ndarray,
                        no_of_clusters: int = 3) -> np.ndarray:
    """
    Colour quantise *rgbImage* using PIL's built-in quantiser.

    Operates on a half-resolution copy for speed.

    Parameters
    ----------
    rgbImage : np.ndarray
        RGB image, dtype uint8.
    no_of_clusters : int
        Number of colours in the output palette (default 3).

    Returns
    -------
    np.ndarray
        Quantised RGB image at original resolution, dtype uint8.
    """
    current_size = np.array(rgbImage.shape)[0:2]
    new_size     = tuple((current_size * 0.5).astype(int))
    resized      = cv2.resize(rgbImage, (new_size[1], new_size[0]))

    im_pil = Image.fromarray(np.uint8(resized))
    im_pil = im_pil.quantize(no_of_clusters, None, 0, None)
    q_img  = np.array(im_pil.convert("RGB"))
    return cv2.resize(q_img, (current_size[1], current_size[0]))